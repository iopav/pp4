"""Reuse successful matching runs without changing existing artifacts."""
import json
from pathlib import Path


# Runtime provenance can differ when a scheduler assigns another GPU/node.
PROVENANCE = {'torch_version', 'device_name', 'source'}


def skip_completed(destination, config):
    destination = Path(destination)
    if not destination.exists():
        return False

    def reject(reason):
        raise FileExistsError(
            f'{destination}: {reason}. Existing results were preserved; '
            'use --output with a new directory to run again.')

    def read(name):
        try:
            value = json.loads((destination / name).read_text(encoding='utf-8'))
            if not isinstance(value, dict):
                raise ValueError('expected JSON object')
            return value
        except (OSError, ValueError) as error:
            reject(f'missing or invalid {name}: {error}')

    status = read('status.json')
    if status.get('status') != 'success':
        reject('run is failed or incomplete, not eligible for skipping')
    previous = read('config.json')
    # Normalize tuples and other JSON-compatible values exactly as config.json.
    current = json.loads(json.dumps(config))
    differences = sorted(key for key in (previous.keys() | current.keys()) - PROVENANCE
                         if key not in previous or key not in current or previous[key] != current[key])
    if differences:
        reject('configuration differs: ' + ', '.join(differences))
    required = ['model.pt', 'metrics.csv']
    if config['method'] in ('gatezo', 'bdnn'):
        required += ['summary.json', 'final_metrics.json']
        if status.get('hidden_neurons') != config['max_hidden']:
            reject('successful status does not match requested hidden size')
        read('summary.json')
        read('final_metrics.json')
    elif status.get('epochs') != config['epochs']:
        reject('successful status does not match requested epochs')
    for name in required:
        path = destination / name
        if not path.is_file() or path.stat().st_size == 0:
            reject(f'missing or empty {name}')
    print(f'[skip] {config["method"]}: completed matching run at {destination}', flush=True)
    return True
