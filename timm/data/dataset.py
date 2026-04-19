""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
from typing import Optional

import torch
import torch.utils.data as data
from PIL import Image

from .readers import create_reader

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 20


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            is_training=False,
            input_img_mode='RGB',
            transform=None,
            target_transform=None,
            additional_features=None,
            **kwargs,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map,
                additional_features=additional_features,
                **kwargs,
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.is_training = is_training
        self.input_img_mode = input_img_mode
        self.transform = transform
        self.target_transform = target_transform
        self.additional_features = additional_features
        self._max_retries = _ERROR_RETRY

    def _open_image(self, img):
        if self.load_bytes:
            return img.read() if hasattr(img, 'read') else img

        if isinstance(img, Image.Image):
            decoded = img.copy()
        else:
            with Image.open(img) as pil_img:
                if self.input_img_mode and pil_img.mode != self.input_img_mode:
                    decoded = pil_img.convert(self.input_img_mode)
                else:
                    decoded = pil_img.copy()

        return decoded

    def _prepare_sample(self, sample):
        img, target, *features = sample
        img = self._open_image(img)

        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        if self.additional_features is None:
            return img, target
        else:
            return img, target, *features

    def _should_retry_error(self, err):
        if hasattr(self.reader, 'retry_on_error'):
            return self.is_training and self.reader.retry_on_error and self.reader.is_retryable_error(err)
        return isinstance(err, (IOError, OSError))

    def _next_retry_index(self, index, attempt):
        if hasattr(self.reader, 'retry_index'):
            return self.reader.retry_index(index, attempt)
        return (index + attempt + 1) % len(self.reader)

    def _log_retry(self, index, err, attempt):
        if hasattr(self.reader, 'log_sample_error'):
            self.reader.log_sample_error(index, err, attempt=attempt)
        else:
            _logger.warning(f'Skipped sample (index {index}). {err}')

    def _load_item_with_retry(self, index, first_error=None):
        current_index = index
        pending_error = first_error

        for attempt in range(self._max_retries):
            if pending_error is None:
                try:
                    return self._prepare_sample(self.reader[current_index])
                except Exception as err:
                    pending_error = err

            if not self._should_retry_error(pending_error):
                raise pending_error

            self._log_retry(current_index, pending_error, attempt)
            current_index = self._next_retry_index(current_index, attempt)
            pending_error = None

        raise RuntimeError(f"Failed to load {self._max_retries} consecutive samples")

    def __getitem__(self, index):
        return self._load_item_with_retry(index)

    def __getitems__(self, indices):
        if not indices:
            return []

        samples = None
        reader_getitems = getattr(self.reader, '__getitems__', None)
        if callable(reader_getitems):
            try:
                samples = reader_getitems(indices)
            except Exception as err:
                if not self._should_retry_error(err):
                    raise

        if samples is None:
            return [self._load_item_with_retry(index) for index in indices]
        if len(samples) != len(indices):
            raise RuntimeError(
                f'Reader batched fetch returned {len(samples)} samples for {len(indices)} indices'
            )

        prepared = []
        for index, sample in zip(indices, samples):
            try:
                prepared.append(self._prepare_sample(sample))
            except Exception as err:
                prepared.append(self._load_item_with_retry(index, first_error=err))
        return prepared

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            is_training=False,
            batch_size=1,
            num_samples=None,
            seed=42,
            repeats=0,
            download=False,
            input_img_mode='RGB',
            input_key=None,
            target_key=None,
            transform=None,
            target_transform=None,
            max_steps=None,
            **kwargs,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                class_map=class_map,
                is_training=is_training,
                batch_size=batch_size,
                num_samples=num_samples,
                seed=seed,
                repeats=repeats,
                download=download,
                input_img_mode=input_img_mode,
                input_key=input_key,
                target_key=target_key,
                max_steps=max_steps,
                **kwargs,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
        if hasattr(self.reader, 'set_loader_cfg'):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __getitems__(self, indices):
        dataset_getitems = getattr(self.dataset, '__getitems__', None)
        if callable(dataset_getitems):
            items = dataset_getitems(indices)
        else:
            items = [self.dataset[i] for i in indices]
        return [self._getitem_from_item(item) for item in items]

    def _getitem_from_item(self, item):
        x, y = item
        x_list = [self._normalize(x)]
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
