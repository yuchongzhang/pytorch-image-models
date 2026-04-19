"""DDP-safe parquet reader for HuggingFace-style image datasets.

This reader is intentionally map-style only. It owns parquet metadata and
indexed row-group access, while DataLoader samplers handle sharding for DDP.
"""
import io
import glob
import logging
import os
from bisect import bisect_right
from collections import OrderedDict
from typing import NamedTuple, Optional, Sequence

try:
    import pyarrow.parquet as pq
    has_pyarrow = True
except ImportError:
    has_pyarrow = False

from .reader import Reader

_logger = logging.getLogger(__name__)

_MAX_OPEN_FILES = int(os.environ.get('TIMM_PARQUET_MAX_OPEN_FILES', 4))
_MAX_LOGGED_ERRORS = int(os.environ.get('TIMM_PARQUET_MAX_LOGGED_ERRORS', 20))
_LOG_EVERY_N_ERRORS = int(os.environ.get('TIMM_PARQUET_LOG_EVERY_N_ERRORS', 100))
_USE_ROW_GROUP_THREADS = bool(int(os.environ.get('TIMM_PARQUET_USE_ROW_GROUP_THREADS', 0)))
_USE_MEMORY_MAP = bool(int(os.environ.get('TIMM_PARQUET_USE_MEMORY_MAP', 0)))


class _RowGroupInfo(NamedTuple):
    file_idx: int
    row_group_idx: int
    start: int
    num_rows: int


class ParquetSampleError(RuntimeError):
    """Context-rich parquet sample loading error."""

    def __init__(
            self,
            message: str,
            *,
            index: Optional[int] = None,
            file_path: Optional[str] = None,
            row_group_idx: Optional[int] = None,
            image_path: Optional[str] = None,
            cause: Optional[BaseException] = None,
    ):
        parts = [message]
        if index is not None:
            parts.append(f'index={index}')
        if file_path:
            parts.append(f'file={file_path}')
        if row_group_idx is not None:
            parts.append(f'row_group={row_group_idx}')
        if image_path:
            parts.append(f'image_path={image_path}')
        if cause is not None:
            parts.append(f'cause={cause}')
        super().__init__(', '.join(parts))
        self.index = index
        self.file_path = file_path
        self.row_group_idx = row_group_idx
        self.image_path = image_path
        self.cause = cause


class ReaderParquet(Reader):
    """Reader for parquet-format image datasets.

    The reader is intentionally map-style only. Parquet-aware distributed
    shuffling is handled by a sampler rather than the reader itself.
    """

    retry_on_error = True
    use_parquet_distributed_sampler = True

    def __init__(
            self,
            root: str,
            split: str = 'train',
            class_map: Optional[dict] = None,
            input_key: str = 'image',
            input_img_mode: str = 'RGB',
            target_key: str = 'label',
            is_training: bool = False,
            batch_size: int = 1,
            num_samples: Optional[int] = None,
            repeats: int = 0,
            seed: int = 42,
            **kwargs,
    ):
        super().__init__()
        assert has_pyarrow, 'Please install pyarrow: pip install pyarrow'

        self.root = root
        self.split = split
        self.image_key = input_key
        self.input_img_mode = input_img_mode
        self.label_key = target_key
        self.is_training = is_training
        self.batch_size = batch_size
        self.repeats = repeats
        self.seed = seed
        self.max_open_files = max(1, _MAX_OPEN_FILES)

        pattern = os.path.join(root, f'{split}-*.parquet')
        self.file_paths = sorted(glob.glob(pattern))
        assert self.file_paths, (
            f'No parquet files found matching {pattern}. '
            f'Expected files like {split}-00000-of-NNNNN.parquet in {root}'
        )
        _logger.info(f'Found {len(self.file_paths)} parquet files for split={split}')

        requested_num_samples = num_samples
        self.total_samples = 0
        self.row_groups = []
        self._row_group_ends = []

        for file_idx, path in enumerate(self.file_paths):
            parquet_file = pq.ParquetFile(path, memory_map=_USE_MEMORY_MAP)
            metadata = parquet_file.metadata
            for row_group_idx in range(parquet_file.num_row_groups):
                row_group_rows = metadata.row_group(row_group_idx).num_rows
                if requested_num_samples is not None and self.total_samples >= requested_num_samples:
                    break
                visible_rows = row_group_rows
                if requested_num_samples is not None:
                    visible_rows = min(visible_rows, requested_num_samples - self.total_samples)
                if visible_rows <= 0:
                    break
                self.row_groups.append(_RowGroupInfo(
                    file_idx=file_idx,
                    row_group_idx=row_group_idx,
                    start=self.total_samples,
                    num_rows=visible_rows,
                ))
                self.total_samples += visible_rows
                self._row_group_ends.append(self.total_samples)
            if requested_num_samples is not None and self.total_samples >= requested_num_samples:
                break

        self.num_samples = self.total_samples

        self.remap_class = False
        if class_map:
            from .class_map import load_class_map
            self.class_to_idx = load_class_map(class_map)
            self.remap_class = True
        else:
            self.class_to_idx = {}

        self._local_pid = None
        self._parquet_files = OrderedDict()
        self._cached_row_group = None
        self._cached_images = None
        self._cached_labels = None
        self._logged_errors = 0

        _logger.info(
            f'Indexed {self.num_samples} samples across {len(self.row_groups)} row groups '
            f'for split={split} (streaming=False)'
        )

    def __len__(self):
        return self.num_samples

    def is_retryable_error(self, err: BaseException):
        return isinstance(err, (ParquetSampleError, OSError))

    def retry_index(self, index: int, attempt: int):
        if not self.num_samples:
            return index
        return (index + attempt + 1) % self.num_samples

    def log_sample_error(self, index: int, err: BaseException, attempt: int = 0):
        self._ensure_local_state()
        self._logged_errors += 1
        should_log = (
            self._logged_errors <= _MAX_LOGGED_ERRORS
            or (
                _LOG_EVERY_N_ERRORS > 0
                and self._logged_errors % _LOG_EVERY_N_ERRORS == 0
            )
        )
        if not should_log:
            return

        context = self.sample_context(index)
        parts = [
            f'Skipping parquet sample during training (attempt {attempt + 1})',
            f'index={index}',
        ]
        if context['file_path']:
            parts.append(f"file={context['file_path']}")
        if context['row_group_idx'] is not None:
            parts.append(f"row_group={context['row_group_idx']}")
        if context['image_path']:
            parts.append(f"image_path={context['image_path']}")
        parts.append(f'error={err}')
        _logger.warning(', '.join(parts))

    def sample_context(self, index: int):
        context = {
            'index': index,
            'file_path': None,
            'row_group_idx': None,
            'image_path': None,
        }
        if index < 0 or index >= self.num_samples:
            return context

        row_group_offset = bisect_right(self._row_group_ends, index)
        row_group = self.row_groups[row_group_offset]
        context['file_path'] = self.file_paths[row_group.file_idx]
        context['row_group_idx'] = row_group.row_group_idx

        try:
            images, _ = self._load_row_group(row_group_offset)
            image_data = images[index - row_group.start].as_py()
            if isinstance(image_data, dict):
                image_path = image_data.get('path')
                if image_path:
                    context['image_path'] = image_path
        except Exception:
            pass
        return context

    def _reset_local_state(self):
        self._parquet_files = OrderedDict()
        self._cached_row_group = None
        self._cached_images = None
        self._cached_labels = None
        self._logged_errors = 0

    def _ensure_local_state(self):
        pid = os.getpid()
        if self._local_pid != pid:
            self._local_pid = pid
            self._reset_local_state()

    def _wrap_error(
            self,
            message: str,
            *,
            index: Optional[int] = None,
            row_group: Optional[_RowGroupInfo] = None,
            image_path: Optional[str] = None,
            cause: Optional[BaseException] = None,
    ):
        file_path = None
        row_group_idx = None
        if row_group is not None:
            file_path = self.file_paths[row_group.file_idx]
            row_group_idx = row_group.row_group_idx
        raise ParquetSampleError(
            message,
            index=index,
            file_path=file_path,
            row_group_idx=row_group_idx,
            image_path=image_path,
            cause=cause,
        ) from cause

    def _get_parquet_file(self, file_idx: int):
        self._ensure_local_state()

        parquet_file = self._parquet_files.get(file_idx)
        if parquet_file is not None:
            self._parquet_files.move_to_end(file_idx)
            return parquet_file

        parquet_file = pq.ParquetFile(self.file_paths[file_idx], memory_map=_USE_MEMORY_MAP)
        self._parquet_files[file_idx] = parquet_file
        if len(self._parquet_files) > self.max_open_files:
            self._parquet_files.popitem(last=False)
        return parquet_file

    def _load_row_group(self, row_group_offset: int):
        self._ensure_local_state()
        if self._cached_row_group == row_group_offset:
            return self._cached_images, self._cached_labels

        row_group = self.row_groups[row_group_offset]
        parquet_file = self._get_parquet_file(row_group.file_idx)
        try:
            table = parquet_file.read_row_group(
                row_group.row_group_idx,
                columns=[self.image_key, self.label_key],
                use_threads=_USE_ROW_GROUP_THREADS,
            )
        except Exception as err:
            self._wrap_error(
                'Failed to read parquet row group',
                index=row_group.start,
                row_group=row_group,
                cause=err,
            )

        self._cached_images = table.column(self.image_key)
        self._cached_labels = table.column(self.label_key)
        self._cached_row_group = row_group_offset
        return self._cached_images, self._cached_labels

    def _resolve_row_group(self, index: int):
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        row_group_offset = bisect_right(self._row_group_ends, index)
        row_group = self.row_groups[row_group_offset]
        return row_group_offset, row_group, index - row_group.start

    def _resolve_image_path(self, image_path: str):
        if os.path.isabs(image_path):
            return image_path
        candidate_path = os.path.join(self.root, image_path)
        if os.path.exists(candidate_path):
            return candidate_path
        return image_path

    def _image_struct_to_bytesio(self, image_data, *, index: int, row_group: _RowGroupInfo):
        if image_data is None:
            self._wrap_error('Image data is missing from parquet sample', index=index, row_group=row_group)

        if isinstance(image_data, dict):
            image_bytes = image_data.get('bytes')
            image_path = image_data.get('path')
            if image_bytes:
                return io.BytesIO(image_bytes)
            if image_path:
                resolved_path = self._resolve_image_path(image_path)
                try:
                    with open(resolved_path, 'rb') as handle:
                        return io.BytesIO(handle.read())
                except Exception as err:
                    self._wrap_error(
                        'Failed to read image path from parquet sample',
                        index=index,
                        row_group=row_group,
                        image_path=image_path,
                        cause=err,
                    )
            self._wrap_error(
                'Image bytes and path are both missing from parquet sample',
                index=index,
                row_group=row_group,
                image_path=image_path,
            )
        if isinstance(image_data, (bytes, bytearray, memoryview)):
            return io.BytesIO(bytes(image_data))

        self._wrap_error(
            f'Unsupported parquet image payload type: {type(image_data).__name__}',
            index=index,
            row_group=row_group,
        )

    def _materialize_sample(self, index: int, row_group_offset: int, row_group: _RowGroupInfo, row_offset: int):
        images, labels = self._load_row_group(row_group_offset)
        try:
            image_data = images[row_offset].as_py()
            label = int(labels[row_offset].as_py())
        except Exception as err:
            self._wrap_error(
                'Failed to materialize parquet sample from row group',
                index=index,
                row_group=row_group,
                cause=err,
            )

        if self.remap_class:
            label = self.class_to_idx[label]
        return self._image_struct_to_bytesio(image_data, index=index, row_group=row_group), label

    def __getitem__(self, index: int):
        row_group_offset, row_group, row_offset = self._resolve_row_group(index)
        return self._materialize_sample(index, row_group_offset, row_group, row_offset)

    def __getitems__(self, indices: Sequence[int]):
        if not indices:
            return []

        requests_by_row_group = OrderedDict()
        for position, index in enumerate(indices):
            row_group_offset, row_group, row_offset = self._resolve_row_group(index)
            requests_by_row_group.setdefault(row_group_offset, []).append((position, index, row_group, row_offset))

        samples = [None] * len(indices)
        for row_group_offset, requests in requests_by_row_group.items():
            images, labels = self._load_row_group(row_group_offset)
            for position, index, row_group, row_offset in requests:
                try:
                    image_data = images[row_offset].as_py()
                    label = int(labels[row_offset].as_py())
                except Exception as err:
                    self._wrap_error(
                        'Failed to materialize parquet sample from batched row group fetch',
                        index=index,
                        row_group=row_group,
                        cause=err,
                    )

                if self.remap_class:
                    label = self.class_to_idx[label]
                samples[position] = (
                    self._image_struct_to_bytesio(image_data, index=index, row_group=row_group),
                    label,
                )
        return samples

    def _filename(self, index, basename=False, absolute=False):
        row_group_offset, row_group, row_offset = self._resolve_row_group(index)
        images, _ = self._load_row_group(row_group_offset)
        image_data = images[row_offset].as_py()
        if isinstance(image_data, dict):
            filename = image_data.get('path', '') or ''
        else:
            filename = ''
        if basename:
            return os.path.basename(filename)
        if absolute and filename:
            return self._resolve_image_path(filename)
        return filename

    def filenames(self, basename=False, absolute=False):
        filename_col = f'{self.image_key}.path'
        names = []
        remaining = self.num_samples
        for path in self.file_paths:
            if remaining <= 0:
                break
            table = pq.read_table(
                path,
                columns=[filename_col],
                memory_map=_USE_MEMORY_MAP,
                use_threads=_USE_ROW_GROUP_THREADS,
            )
            file_names = table.column(0).to_pylist()
            names.extend(file_names[:remaining])
            remaining -= len(file_names)

        if basename:
            return [os.path.basename(name) for name in names]
        if absolute:
            return [self._resolve_image_path(name) if name else name for name in names]
        return names
