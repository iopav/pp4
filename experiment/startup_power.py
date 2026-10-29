"""Print pre-training board power once per launcher invocation (not idle power)."""
import argparse
import json
import os
import time

MARKER = 'TEXTURE_STARTUP_POWER_REPORTED'


def report_startup_power(device, seconds=3.0, interval=0.1):
    if os.environ.get(MARKER) == '1':
        return
    result = dict(measurement='pre_training_board_power', power_w=None,
                  sampling_seconds=seconds, samples=0)
    nv = None
    try:
        if str(device).startswith('cpu'):
            raise RuntimeError('CPU run: GPU telemetry unavailable')
        import torch
        import pynvml
        nv = pynvml
        nv.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(torch.device(device)).uuid)
        if not uuid.startswith(('GPU-', 'MIG-')):
            uuid = 'GPU-' + uuid
        handle = nv.nvmlDeviceGetHandleByUUID(uuid)
        values = []
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            values.append(nv.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            time.sleep(min(interval, max(0, end - time.perf_counter())))
        if not values:
            raise RuntimeError('No power samples')
        result.update(power_w=sum(values) / len(values), samples=len(values))
    except Exception as error:
        result['reason'] = f'{type(error).__name__}: {error}'
    finally:
        if nv is not None:
            try:
                nv.nvmlShutdown()
            except Exception:
                pass
    print('[startup_power] ' + json.dumps(result), flush=True)
    os.environ[MARKER] = '1'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    report_startup_power(parser.parse_args().device)
