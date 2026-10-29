"""Requested accounting: epoch-end optimizer state + epoch BP saved peak.

This sum is not a simultaneous allocation peak or full checkpoint size.
RNG, sampler state, configuration and parameter groups are excluded.
"""
from collections.abc import Mapping
import numpy as np
import torch
from bp_memory import storage_identity

AUX_MEMORY_VERSION = 'optimizer_state_plus_epoch_bp_peak_v1'


class SynchronizedSegment:
    """Preserve timing boundaries without measuring the obsolete CUDA peak metric."""
    def __init__(self, device):
        self.device = torch.device(device)

    def __enter__(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return self

    def __exit__(self, *exc):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)


def optimizer_state_bytes(optimizer):
    seen = set()

    def count(value):
        if isinstance(value, torch.Tensor):
            key = ('tensor', *storage_identity(value))
            size = value.untyped_storage().nbytes()
        elif isinstance(value, np.ndarray):
            root = value
            while isinstance(root.base, np.ndarray):
                root = root.base
            key, size = ('numpy', id(root)), root.nbytes
        elif isinstance(value, Mapping):
            return sum(count(item) for item in value.values())
        elif isinstance(value, (list, tuple)):
            return sum(count(item) for item in value)
        elif isinstance(value, (bool, int, float, np.number)):
            # Logical metadata bytes, not Python object heap size.
            return 8
        else:
            return 0
        if key in seen:
            return 0
        seen.add(key)
        return int(size)

    return count(optimizer.state)


def make_optimizer(parameters, config):
    if config['optimizer'] == 'SGD':
        return torch.optim.SGD(parameters, lr=config['lr'], momentum=config['momentum'])
    if config['optimizer'] == 'Adam':
        return torch.optim.Adam(parameters, lr=config['lr'])
    raise ValueError('Supported optimizers: Adam, SGD')
