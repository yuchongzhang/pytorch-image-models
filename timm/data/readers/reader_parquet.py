"""Memory-efficient parquet reader for HuggingFace-style image datasets.

This reader keeps only parquet metadata in memory at initialization. It supports:

* Lazy random access with a one-row-group cache for validation / inference.
* Streaming iteration for training that shuffles shards and row groups, but only
  materializes one row group at a time per worker.

This avoids the previous behaviour of loading every compressed image into RAM up
front, which quickly OOMs multi-rank ImageNet training jobs.
"""
import io
import glob
import logging
import math
import os
import random
from bisect import bisect_right
from collections import OrderedDict
from typing import NamedTuple, Optional

from PIL import Image
import torch
import torch.distributed as dist

try:
    import pyarrow.parquet as pq
    has_pyarrow = True
except ImportError:
    has_pyarrow = False

from .reader import Reader
from .shared_count import SharedCount

_logger = logging.getLogger(__name__)

_MAX_OPEN_FILES = int(os.environ.get('TIMM_PARQUET_MAX_OPEN_FILES', 4))
_USE_ROW_GROUP_THREADS = bool(int(os.environ.get('TIMM_PARQUET_USE_ROW_GROUP_THREADS', 0)))


class _RowGroupInfo(NamedTuple):
    file_idx: int
    row_group_idx: int
    start: int
    num_rows: int


class ReaderParquet(Reader):
    """Reader for parquet-format image datasets."""

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
        self.common_seed = seed
        self.max_open_files = max(1, _MAX_OPEN_FILES)

        pattern = os.path.join(root, f'{split}-*.parquet')
        self.file_paths = sorted(glob.glob(pattern))
        assert self.file_paths, (
            f'No parquet files found matching {pattern}. '
            f'Expected files like {split}-00000-of-NNNNN.parquet in {root}'
        )
        _logger.info(f'Found {len(self.file_paths)} parquet files for split={split}')

        self.file_num_rows = []
        self.file_num_row_groups = []
        self.row_groups = []
        self._row_group_ends = []
        total_samples = 0
        for file_idx, path in enumerate(self.file_paths):
            parquet_file = pq.ParquetFile(path)
            self.file_num_rows.append(parquet_file.metadata.num_rows)
            self.file_num_row_groups.append(parquet_file.num_row_groups)
            for row_group_idx in range(parquet_file.num_row_groups):
                row_group_rows = parquet_file.metadata.row_group(row_group_idx).num_rows
                total_samples += row_group_rows
                self.row_groups.append(_RowGroupInfo(
                    file_idx=file_idx,
                    row_group_idx=row_group_idx,
                    start=total_samples - row_group_rows,
                    num_rows=row_group_rows,
                ))
                self._row_group_ends.append(total_samples)

        self.total_samples = total_samples
        self.num_samples = min(num_samples, total_samples) if num_samples is not None else total_samples

        self.remap_class = False
        if class_map:
            from .class_map import load_class_map
            self.class_to_idx = load_class_map(class_map)
            self.remap_class = True
        else:
            self.class_to_idx = {}

        # Distributed world state
        self.dist_rank = 0
        self.dist_num_replicas = 1
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            self.dist_rank = dist.get_rank()
            self.dist_num_replicas = dist.get_world_size()

        # Worker / process-local state, initialized lazily.
        self.worker_info = None
        self.worker_id = 0
        self.num_workers = 1
        self.global_worker_id = 0
        self.global_num_workers = 1
        self.epoch = SharedCount()
        self._parquet_files = OrderedDict()
        self._cached_row_group = None
        self._cached_images = None
        self._cached_labels = None

        _logger.info(
            f'Indexed {self.num_samples} samples across {len(self.row_groups)} row groups '
            f'for split={split} (streaming={self.is_training})'
        )

    def set_epoch(self, count):
        self.epoch.value = count

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        if num_workers is not None:
            self.num_workers = num_workers
            self.global_num_workers = self.dist_num_replicas * self.num_workers

    def _lazy_init_worker(self):
        if self.worker_info is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                self.worker_info = worker_info
                self.worker_id = worker_info.id
                self.num_workers = worker_info.num_workers
            self.global_num_workers = self.dist_num_replicas * self.num_workers
            self.global_worker_id = self.dist_rank * self.num_workers + self.worker_id

    def _num_samples_per_worker(self):
        num_worker_samples = max(1, self.repeats) * self.num_samples / max(
            self.global_num_workers,
            self.dist_num_replicas,
        )
        if self.is_training or self.dist_num_replicas > 1:
            num_worker_samples = math.ceil(num_worker_samples)
        if self.is_training and self.batch_size is not None:
            num_worker_samples = math.ceil(num_worker_samples / self.batch_size) * self.batch_size
        return int(num_worker_samples)

    def _get_parquet_file(self, file_idx: int):
        parquet_file = self._parquet_files.get(file_idx)
        if parquet_file is not None:
            self._parquet_files.move_to_end(file_idx)
            return parquet_file

        parquet_file = pq.ParquetFile(self.file_paths[file_idx], memory_map=True)
        self._parquet_files[file_idx] = parquet_file
        if len(self._parquet_files) > self.max_open_files:
            self._parquet_files.popitem(last=False)
        return parquet_file

    def _load_row_group(self, row_group: _RowGroupInfo):
        parquet_file = self._get_parquet_file(row_group.file_idx)
        # PyTorch already parallelizes parquet reads across DataLoader workers.
        # Keep Arrow's internal thread pool disabled by default to avoid nested
        # worker x Arrow oversubscription and worker stalls.
        table = parquet_file.read_row_group(
            row_group.row_group_idx,
            columns=[self.image_key, self.label_key],
            use_threads=_USE_ROW_GROUP_THREADS,
        )
        return table.column(self.image_key), table.column(self.label_key)

    def _resolve_image_path(self, image_path: str):
        if os.path.isabs(image_path):
            return image_path
        candidate_path = os.path.join(self.root, image_path)
        if os.path.exists(candidate_path):
            return candidate_path
        return image_path

    def _image_struct_to_bytesio(self, image_data):
        if image_data is None:
            raise RuntimeError('Image data is missing from parquet sample.')

        if isinstance(image_data, dict):
            image_bytes = image_data.get('bytes')
            if image_bytes:
                return io.BytesIO(image_bytes)
            image_path = image_data.get('path')
            if image_path:
                return open(self._resolve_image_path(image_path), 'rb')
        elif isinstance(image_data, (bytes, bytearray, memoryview)):
            return io.BytesIO(bytes(image_data))

        raise RuntimeError('Unsupported parquet image payload.')

    def _decode_image(self, image_data):
        if isinstance(image_data, dict):
            image_bytes = image_data.get('bytes')
            if image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
                image.load()
            else:
                image_path = image_data.get('path')
                if not image_path:
                    raise RuntimeError('Image path is missing from parquet sample.')
                with open(self._resolve_image_path(image_path), 'rb') as handle:
                    image = Image.open(handle)
                    image.load()
        elif isinstance(image_data, (bytes, bytearray, memoryview)):
            image = Image.open(io.BytesIO(bytes(image_data)))
            image.load()
        else:
            raise RuntimeError('Unsupported parquet image payload.')

        if self.input_img_mode and image.mode != self.input_img_mode:
            image = image.convert(self.input_img_mode)
        return image

    def _get_row_group_for_index(self, index: int):
        row_group_offset = bisect_right(self._row_group_ends, index)
        row_group = self.row_groups[row_group_offset]
        if self._cached_row_group != row_group_offset:
            self._cached_images, self._cached_labels = self._load_row_group(row_group)
            self._cached_row_group = row_group_offset
        return row_group, index - row_group.start

    def __getitem__(self, index):
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)

        row_group, row_offset = self._get_row_group_for_index(index)
        image_data = self._cached_images[row_offset].as_py()
        label = int(self._cached_labels[row_offset].as_py())
        if self.remap_class:
            label = self.class_to_idx[label]
        return self._image_struct_to_bytesio(image_data), label

    def __iter__(self):
        self._lazy_init_worker()

        row_group_indices = list(range(len(self.row_groups)))
        if self.is_training:
            shuffle_rng = random.Random(self.common_seed + self.epoch.value)
            shuffle_rng.shuffle(row_group_indices)

        if self.global_num_workers > 1:
            assigned_row_groups = row_group_indices[self.global_worker_id::self.global_num_workers]
            if not assigned_row_groups:
                assigned_row_groups = [row_group_indices[self.global_worker_id % len(row_group_indices)]]
        else:
            assigned_row_groups = row_group_indices

        target_sample_count = self._num_samples_per_worker()
        yielded = 0
        cycle = 0

        while True:
            cycle_seed = self.common_seed + self.epoch.value + cycle * 104729 + self.global_worker_id
            cycle_rng = random.Random(cycle_seed)
            cycle_row_groups = list(assigned_row_groups)
            if self.is_training:
                cycle_rng.shuffle(cycle_row_groups)

            for row_group_idx in cycle_row_groups:
                row_group = self.row_groups[row_group_idx]
                images, labels = self._load_row_group(row_group)
                row_order = list(range(len(images)))
                if self.is_training:
                    cycle_rng.shuffle(row_order)

                for row_idx in row_order:
                    if yielded >= target_sample_count:
                        return
                    image = self._decode_image(images[row_idx].as_py())
                    label = int(labels[row_idx].as_py())
                    if self.remap_class:
                        label = self.class_to_idx[label]
                    yield image, label
                    yielded += 1

            if not self.is_training:
                return
            cycle += 1

    def __len__(self):
        if self.is_training:
            return self._num_samples_per_worker() * self.num_workers
        return self.num_samples

    def _filename(self, index, basename=False, absolute=False):
        row_group, row_offset = self._get_row_group_for_index(index)
        image_data = self._cached_images[row_offset].as_py()
        if isinstance(image_data, dict):
            filename = image_data.get('path', '')
        else:
            filename = ''
        if basename:
            filename = os.path.basename(filename)
        elif absolute and filename:
            filename = self._resolve_image_path(filename)
        return filename

    def filenames(self, basename=False, absolute=False):
        filename_col = f'{self.image_key}.path'
        names = []
        for path in self.file_paths:
            table = pq.read_table(path, columns=[filename_col])
            names.extend(table.column(0).to_pylist())

        names = names[:self.num_samples]

        if basename:
            return [os.path.basename(name) for name in names]
        if absolute:
            return [self._resolve_image_path(name) for name in names]
        return names
