"""Dataset adapters exposing float32 (time, channels) inputs and integer labels."""
from pathlib import Path
import math
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def default_data_root(dataset='texture-mini'):
    return Path(__file__).resolve().parents[1] / 'data' / dataset


DEFAULT_DATA_ROOT = default_data_root()


def add_data_arguments(parser):
    parser.add_argument('--dataset', choices=('texture-mini', 'shd', 'texture'), default='texture-mini')
    parser.add_argument('--data-root', type=Path, help='Dataset directory (or explicit NPZ file)')
    parser.add_argument('--shd-dt', type=float, default=0.005, help='SHD bin width in seconds')
    parser.add_argument('--shd-max-time', type=float, default=1.4, help='Exclusive time cutoff in seconds')
    parser.add_argument('--shd-channel-group-size', type=int, default=1)


def load_data(args, smoke=False):
    args.data_root = args.data_root or default_data_root(args.dataset)
    return datasets(args.data_root, smoke, args.dataset, args.shd_dt,
                    args.shd_max_time, args.shd_channel_group_size)


def labels(y, classes):
    y = np.asarray(y)
    if y.ndim == 2:
        if y.shape[1] != classes or not np.isin(y, (0, 1)).all() or not (y.sum(1) == 1).all():
            raise ValueError('Expected strict one-hot labels with one active class per row')
        y = y.argmax(1)
    if (y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer)
            or y.min() < 0 or y.max() >= classes):
        raise ValueError(f'Expected integer labels in 0..{classes - 1}')
    return y.astype(np.int64, copy=False)


class Sequences(Dataset):
    def __init__(self, x, y, classes, smoke=False):
        self.x, self.y = x, labels(y, classes)
        if x.ndim != 3 or x.shape[0] != len(self.y) or min(x.shape[1:]) < 1:
            raise ValueError(f'Expected nonempty (N,T,C) inputs matching labels, got {x.shape}')
        self.steps = min(16, x.shape[1]) if smoke else x.shape[1]
        self.inputs, self.classes = x.shape[2], classes
        self.count = min(8, len(self.y)) if smoke else len(self.y)

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        x = np.array(self.x[i, :self.steps], dtype=np.float32, copy=True)
        if not np.isfinite(x).all():
            raise ValueError(f'Nonfinite input at sample {i}')
        return torch.from_numpy(x), int(self.y[i])


class SHD(Dataset):
    """Bin HDF5 events to binary occupancy; no augmentation or normalization."""
    def __init__(self, path, dt, max_time, group, smoke=False):
        import h5py
        if not math.isfinite(dt) or not math.isfinite(max_time) or dt <= 0 or max_time <= 0 or group < 1:
            raise ValueError('SHD dt, max_time and channel group size must be positive and finite')
        self.path, self.dt, self.max_time, self.group = Path(path), dt, max_time, group
        with h5py.File(path, 'r') as f:
            self.y = labels(f['labels'][:], 20)
            if len(f['spikes/times']) != len(self.y) or len(f['spikes/units']) != len(self.y):
                raise ValueError('SHD event arrays and labels have different lengths')
        full_steps = math.ceil(max_time / dt)
        self.steps = min(16, full_steps) if smoke else full_steps
        self.inputs, self.classes = math.ceil(700 / group), 20
        self.count = min(8, len(self.y)) if smoke else len(self.y)

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        import h5py
        with h5py.File(self.path, 'r') as f:
            times = np.asarray(f['spikes/times'][i], dtype=np.float64)
            units = np.asarray(f['spikes/units'][i])
        if (times.shape != units.shape or times.ndim != 1 or not np.isfinite(times).all()
                or not np.issubdtype(units.dtype, np.integer)
                or (units < 0).any() or (units >= 700).any() or (times < 0).any()):
            raise ValueError(f'Invalid SHD events at sample {i}')
        keep = times < min(self.max_time, self.steps * self.dt)
        bins = np.floor(times[keep] / self.dt).astype(np.int64)
        x = np.zeros((self.steps, self.inputs), dtype=np.float32)
        x[bins, units[keep] // self.group] = 1
        return torch.from_numpy(x), int(self.y[i])


def datasets(root, smoke=False, dataset='texture-mini', shd_dt=0.005,
             shd_max_time=1.4, shd_channel_group_size=1):
    root = Path(root)
    if dataset == 'shd':
        pair = tuple(SHD(root / f'shd_{part}.h5', shd_dt, shd_max_time,
                         shd_channel_group_size, smoke) for part in ('train', 'test'))
    elif dataset == 'texture-mini':
        path = root if root.is_file() else root / 'texture_spikesT450_T77.npz'
        with np.load(path, allow_pickle=False) as source:
            pair = tuple(Sequences(source[f'X_{part}'], source[f'Y_{part}'], 12, smoke)
                         for part in ('train', 'test'))
    elif dataset == 'texture':
        def read(part):
            if (root / f'X_{part}_texture.npy').exists():
                return [np.load(root / f'{key}_{part}_texture.npy', mmap_mode='r', allow_pickle=False)
                        for key in ('X', 'Y')]
            return [np.concatenate([np.load(root / f'{age}6classes/{key}_{part}_6cla_{age}.npy',
                                            allow_pickle=False) for age in ('old', 'new')])
                    for key in ('X', 'Y')]
        pair = tuple(Sequences(*read(part), 12, smoke) for part in ('train', 'test'))
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')
    if (pair[0].steps, pair[0].inputs) != (pair[1].steps, pair[1].inputs):
        raise ValueError('Train/test time and channel dimensions differ')
    return pair


def data_metadata(args, train):
    result = dict(dataset=args.dataset, steps=train.steps, inputs=train.inputs, classes=train.classes)
    if args.dataset == 'shd':
        result.update(shd_dt=args.shd_dt, shd_max_time=args.shd_max_time,
                      shd_channel_group_size=args.shd_channel_group_size,
                      shd_encoding='binary_occupancy', shd_augmentation=False)
    return result
