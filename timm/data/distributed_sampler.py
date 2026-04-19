import math
import random
import torch
from torch.utils.data import Sampler
import torch.distributed as dist


class OrderedDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


class RepeatAugSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU). Heavily based on torch.utils.data.DistributedSampler

    This sampler was taken from https://github.com/facebookresearch/deit/blob/0c4b8f60/samplers.py
    Used in
    Copyright (c) 2015-present, Facebook, Inc.
    """

    def __init__(
            self,
            dataset,
            num_replicas=None,
            rank=None,
            shuffle=True,
            num_repeats=3,
            selected_round=256,
            selected_ratio=0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # Determine the number of samples to select per epoch for each rank.
        # num_selected logic defaults to be the same as original RASampler impl, but this one can be tweaked
        # via selected_ratio and selected_round args.
        selected_ratio = selected_ratio or num_replicas  # ratio to reduce selected samples by, num_replicas if 0
        if selected_round:
            self.num_selected_samples = int(math.floor(
                 len(self.dataset) // selected_round * selected_round / selected_ratio))
        else:
            self.num_selected_samples = int(math.ceil(len(self.dataset) / selected_ratio))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # produce repeats e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2....]
        if isinstance(self.num_repeats, float) and not self.num_repeats.is_integer():
            # resample for repeats w/ non-integer ratio
            repeat_size = math.ceil(self.num_repeats * len(self.dataset))
            indices = indices[torch.tensor([int(i // self.num_repeats) for i in range(repeat_size)])]
        else:
            indices = torch.repeat_interleave(indices, repeats=int(self.num_repeats), dim=0)
        indices = indices.tolist()  # leaving as tensor thrashes dataloader memory
        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample per rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # return up to num selected samples
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _unwrap_dataset_reader(dataset):
    current = dataset
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        reader = getattr(current, 'reader', None)
        if reader is not None:
            return reader
        current = getattr(current, 'dataset', None)
    return None


class ParquetDistributedSampler(Sampler):
    """Distributed sampler that preserves parquet row-group locality."""

    def __init__(
            self,
            dataset,
            batch_size,
            num_replicas=None,
            rank=None,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if batch_size <= 0:
            raise ValueError('batch_size must be positive for ParquetDistributedSampler')

        self.dataset = dataset
        self.reader = _unwrap_dataset_reader(dataset)
        if self.reader is None or not getattr(self.reader, 'use_parquet_distributed_sampler', False):
            raise ValueError('ParquetDistributedSampler requires a parquet-backed dataset')

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = self._compute_num_samples()
        self.total_size = self.num_samples * self.num_replicas

    def _repeat_factor(self):
        return float(max(1, getattr(self.reader, 'repeats', 0)))

    def _repeat_indices(self, indices):
        repeats = self._repeat_factor()
        if repeats <= 1:
            return indices
        if repeats.is_integer():
            return indices * int(repeats)
        repeat_size = math.ceil(len(indices) * repeats)
        return [indices[int(i // repeats)] for i in range(repeat_size)]

    def _build_epoch_indices(self):
        rng = random.Random(getattr(self.reader, 'seed', 0) + self.epoch)
        row_group_order = list(range(len(self.reader.row_groups)))
        rng.shuffle(row_group_order)

        indices = []
        for row_group_idx in row_group_order:
            row_group = self.reader.row_groups[row_group_idx]
            row_group_indices = list(range(row_group.start, row_group.start + row_group.num_rows))
            rng.shuffle(row_group_indices)
            indices.extend(row_group_indices)
        return self._repeat_indices(indices)

    def _compute_num_samples(self):
        base_length = len(self.dataset)
        if base_length <= 0:
            return 0

        repeats = self._repeat_factor()
        if repeats <= 1:
            repeated_length = base_length
        elif repeats.is_integer():
            repeated_length = base_length * int(repeats)
        else:
            repeated_length = math.ceil(base_length * repeats)

        block = self.num_replicas * self.batch_size
        usable_length = repeated_length - (repeated_length % block)
        return usable_length // self.num_replicas

    def __iter__(self):
        if self.num_samples <= 0:
            return iter([])

        indices = self._build_epoch_indices()
        indices = indices[:self.total_size]
        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(indices[start:end])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
