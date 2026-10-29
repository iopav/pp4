"""Whole-device GPU energy estimates over explicitly bounded windows.

Standalone Texture implementation; PP3 uses a separate counter-difference meter.
Counter updates can straddle short phase boundaries; power integration avoids
assigning a counter jump to one epoch, but remains limited by sensor resolution.
"""
import math
import threading
import time
import torch


ENERGY_MEASUREMENT_VERSION = 'window_power_integral_v1'


class TotalEnergyCounter:
    """Synchronized cumulative mJ readings; never fall back to power estimates.

    A missing value means unsupported/unavailable telemetry. Equal consecutive
    readings remain valid counter snapshots, not claims of zero window energy.
    """

    def __init__(self, device):
        self.device = device
        self.nv = None
        self.previous = None
        self.reason = 'CPU run: GPU energy unavailable'
        if device.type != 'cuda':
            return
        try:
            import pynvml as nv
            nv.nvmlInit()
            self.nv = nv
            uuid = str(torch.cuda.get_device_properties(device).uuid)
            if not uuid.startswith(('GPU-', 'MIG-')):
                uuid = 'GPU-' + uuid
            self.handle = nv.nvmlDeviceGetHandleByUUID(uuid)
            self.reason = ''
        except Exception as error:
            self.reason = f'{type(error).__name__}: {error}'

    def read(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        if self.reason or self.nv is None:
            return time.perf_counter(), None
        try:
            value = self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            if not isinstance(value, int) or value < 0:
                raise ValueError('Invalid GPU cumulative energy counter')
            if self.previous is not None and value < self.previous:
                raise ValueError('GPU energy counter reset')
            self.previous = value
            return time.perf_counter(), value
        except Exception as error:
            self.reason = f'{type(error).__name__}: {error}'
            return time.perf_counter(), None

    def close(self):
        if self.nv is not None:
            nv, self.nv = self.nv, None
            nv.nvmlShutdown()


class CounterWindowAllocator:
    """Allocate contiguous counter windows to epochs by their elapsed time.

    Callers supply synchronized (seconds, mJ) boundaries. Short epochs are
    grouped until min_seconds; long epochs are kept separate. A final partial
    group is allocated only if its counter advanced, never filled with zero.
    No assumptions about which operations fall between boundaries are made here.
    """

    def __init__(self, min_seconds=1.0):
        if not math.isfinite(min_seconds) or min_seconds <= 0:
            raise ValueError('Minimum energy window must be finite and positive')
        self.min_seconds = min_seconds
        self.pending = []
        self.allocations = {}

    def add(self, key, start, end):
        if key in self.allocations or any(item[0] == key for item in self.pending):
            raise ValueError('Duplicate energy window key')
        seconds = end[0] - start[0]
        if seconds <= 0:
            raise ValueError('Energy window duration must be positive')
        if seconds >= self.min_seconds:
            # Do not absorb a long epoch into an earlier short-epoch group.
            self.flush()
            self.pending.append((key, start, end))
            self.flush()
        else:
            self.pending.append((key, start, end))
            span = end[0] - self.pending[0][1][0]
            initial = self.pending[0][1][1]
            valid = initial is not None and end[1] is not None and end[1] > initial
            if span >= self.min_seconds and valid:
                self.flush()
        return self.allocations

    def flush(self):
        if not self.pending:
            return self.allocations
        initial, final = self.pending[0][1][1], self.pending[-1][2][1]
        values = [boundary[1] for _, start, end in self.pending for boundary in (start, end)]
        valid = (all(value is not None for value in values)
                 and all(b >= a for a, b in zip(values, values[1:]))
                 and final > initial)
        energy = (final - initial) / 1000 if valid else None
        durations = [end[0] - start[0] for _, start, end in self.pending]
        total_seconds = sum(durations)
        assigned = 0.
        for index, ((key, start, end), seconds) in enumerate(zip(self.pending, durations)):
            joules = None
            if energy is not None:
                joules = (energy - assigned if index == len(self.pending) - 1
                          else energy * seconds / total_seconds)
                assigned += joules
            self.allocations[key] = dict(seconds=seconds, gpu_energy_j=joules)
        self.pending.clear()
        return self.allocations


class Energy:
    NOTE = ('Estimated from whole-GPU NVML power samples; sensor readings may be '
            'averaged or delayed. Short-window energy is not independently '
            'resolved; other GPU activity and idle power are included.')

    def __init__(self, device, interval=.1):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('Power sample interval must be finite and positive')
        self.device, self.interval = device, interval
        self.reason = 'CPU run: GPU energy unavailable'
        self.mode = 'unavailable'
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = self.nv = None
        self.joules = None
        if device.type != 'cuda':
            return
        try:
            import pynvml as nv
            nv.nvmlInit()
            self.nv = nv
            uuid = str(torch.cuda.get_device_properties(device).uuid)
            if not uuid.startswith(('GPU-', 'MIG-')):
                uuid = 'GPU-' + uuid
            self.handle = nv.nvmlDeviceGetHandleByUUID(uuid)
            self.uuid = uuid
            self.power_source = 'nvmlDeviceGetPowerUsage (may be time-averaged)'
            self.read_milliwatts = lambda: nv.nvmlDeviceGetPowerUsage(self.handle)
            # Prefer explicitly instantaneous telemetry to the one-second average
            # returned by GetPowerUsage on some architectures. Unsupported devices
            # retain the documented, explicitly labelled averaged-power estimate.
            try:
                self.instant_power()
                self.read_milliwatts = self.instant_power
                self.power_source = 'NVML_FI_DEV_POWER_INSTANT'
            except Exception:
                pass
            self.mode = 'power_integral_estimate'
            self.reason = ''
        except Exception as error:
            self.reason = f'{type(error).__name__}: {error}'

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def instant_power(self):
        nv = self.nv
        value = nv.nvmlDeviceGetFieldValues(
            self.handle, [nv.NVML_FI_DEV_POWER_INSTANT])[0]
        if value.nvmlReturn != nv.NVML_SUCCESS:
            raise RuntimeError('Instantaneous GPU power telemetry unavailable')
        if value.valueType != nv.NVML_VALUE_TYPE_UNSIGNED_INT:
            raise ValueError('Unexpected instantaneous GPU power value type')
        return value.value.uiVal

    def sample(self):
        try:
            watts = self.read_milliwatts() / 1000
            timestamp = time.perf_counter()
            if not math.isfinite(watts) or watts < 0:
                raise ValueError('Invalid GPU power reading')
            self.samples.append((timestamp, watts))
            return timestamp
        except Exception as error:
            self.reason = f'{type(error).__name__}: {error}'
            return time.perf_counter()

    def poll(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        try:
            self.synchronize()
            self.start = time.perf_counter()
            if self.mode == 'power_integral_estimate':
                # Boundaries are sampled even for sub-poll-period epochs.
                self.start = self.sample()
                if not self.reason:
                    self.thread = threading.Thread(target=self.poll, daemon=True)
                    self.thread.start()
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        self.stop_event.set()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join()
        if self.nv is not None:
            nv, self.nv = self.nv, None
            nv.nvmlShutdown()

    def __exit__(self, *exc):
        try:
            self.synchronize()
            # End before joining the sampler; shutdown is not training time.
            self.end = (self.sample() if self.mode == 'power_integral_estimate'
                        else time.perf_counter())
            self.seconds = self.end - self.start
        finally:
            self.close()
        if self.mode == 'power_integral_estimate' and not self.reason:
            points = sorted((t, p) for t, p in self.samples
                            if self.start <= t <= self.end)
            if self.seconds <= 0 or len(points) < 2:
                self.reason = 'Insufficient power samples or nonpositive measurement duration'
                return
            self.joules = sum((b[0] - a[0]) * (a[1] + b[1]) / 2
                              for a, b in zip(points, points[1:]))

    def result(self):
        return dict(seconds=self.seconds, gpu_energy_j=self.joules,
                    energy_method=self.mode,
                    energy_reason=self.reason or (self.NOTE + ' Source: ' + self.power_source),
                    gpu_uuid=getattr(self, 'uuid', None))


class AuxiliaryMemory:
    """Peak live Torch device memory other than resident model parameters."""

    def __init__(self, device, parameter_device_bytes=0):
        self.device = device
        self.parameter_device_bytes = int(parameter_device_bytes)
        self.peak_device_allocated_bytes = None
        self.auxiliary_mem_bytes = None
        self.method = 'unavailable'
        self.reason = 'CPU run: CUDA peak allocation is unavailable'

    def __enter__(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.method = 'torch_cuda_max_memory_allocated_minus_parameters'
            self.reason = ''
        return self

    def __exit__(self, *exc):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            self.peak_device_allocated_bytes = int(
                torch.cuda.max_memory_allocated(self.device)
            )
            if self.peak_device_allocated_bytes < self.parameter_device_bytes:
                raise RuntimeError(
                    'Peak device allocation is smaller than resident parameter storage'
                )
            self.auxiliary_mem_bytes = (
                self.peak_device_allocated_bytes - self.parameter_device_bytes
            )

    def result(self):
        return dict(
            auxiliary_mem_bytes=self.auxiliary_mem_bytes,
            auxiliary_mem_method=self.method,
            auxiliary_mem_reason=self.reason,
        )
