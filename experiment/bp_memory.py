"""Live storage retained by autograd for backward, excluding model parameters.

Counts unique underlying tensor storages on the selected device, not graph-node
metadata, allocation overhead, gradients, or all backward workspaces. Saved
inputs/labels are included when autograd retains them. A view retains its full
backing storage. Hooks keep detached aliases (no clone/offload); packed objects
release accounting when autograd releases them, not on the first unpack.
"""
import torch


BP_MEMORY_VERSION = 'saved_nonparameter_storage_peak_v1'
BP_MEMORY_FIELD = 'hiddenstatememfor bp'


def storage_identity(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes())


class _Saved:
    __slots__ = ('tensor', 'owner', 'key')

    def __init__(self, tensor, owner, key):
        # Do not retain a graph output with its grad_fn: this forms a cycle.
        self.tensor = tensor.detach()
        self.owner, self.key = owner, key
        if key is not None:
            owner._acquire(key)

    def __del__(self):
        if self.key is not None:
            self.owner._release(self.key)


class SavedTensorMemory:
    """Measure one forward/loss/backward batch; epoch metric is max over batches."""

    def __init__(self, parameters, device):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.excluded = {storage_identity(p) for p in parameters}
        self.references = {}
        self.current_bytes = self.peak_bytes = 0

    def _acquire(self, key):
        if key not in self.references:
            self.references[key] = 0
            self.current_bytes += key[2]
            self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        self.references[key] += 1

    def _release(self, key):
        self.references[key] -= 1
        if self.references[key] == 0:
            self.current_bytes -= key[2]
            del self.references[key]

    def _pack(self, tensor):
        key = storage_identity(tensor) if tensor.device == self.device else None
        if key in self.excluded:
            key = None
        return _Saved(tensor, self, key)

    @staticmethod
    def _unpack(saved):
        return saved.tensor

    def __enter__(self):
        self.hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self.hooks.__enter__()
        return self

    def __exit__(self, *exc):
        return self.hooks.__exit__(*exc)
