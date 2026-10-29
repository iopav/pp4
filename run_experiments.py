"""PP4: shared architecture controls and native TP/ESPP/SNNFF/SPSN rules."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BACKBONE = dict(hidden_size=49, layers=1)
METHODS = ('tp', 'espp', 'snnff', 'spsn')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=(*METHODS, 'ffsnn'), default=METHODS)
    parser.add_argument('--hidden-size', type=int, default=BACKBONE['hidden_size'])
    parser.add_argument('--layers', type=int, default=BACKBONE['layers'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data'/'texture-mini')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    parser.add_argument('--gpu')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if min(args.hidden_size, args.layers, args.epochs, args.batch_size) < 1:
        parser.error('architecture and training dimensions must be positive')
    if args.cpu and args.gpu is not None:
        parser.error('--cpu and --gpu are mutually exclusive')
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu = args.gpu if args.gpu is not None else (inherited.split(',')[0] if inherited else '0')
    if not args.cpu and (not gpu or gpu == '-1' or ',' in gpu):
        parser.error('Select exactly one GPU')
    if not args.cpu and inherited is not None and gpu not in inherited.split(','):
        parser.error('GPU must belong to inherited allocation')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='' if args.cpu else gpu,
               PYTHONHASHSEED=str(args.seed), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    device = 'cpu' if args.cpu else 'cuda:0'
    if not args.dry_run:
        env.pop('TEXTURE_STARTUP_POWER_REPORTED', None)
        subprocess.run([sys.executable, str(ROOT/'experiment/startup_power.py'), '--device', device], env=env, check=True)
        env['TEXTURE_STARTUP_POWER_REPORTED'] = '1'
    for method in dict.fromkeys('snnff' if m == 'ffsnn' else m for m in args.methods):
        command = [sys.executable, str(ROOT/'experiment/train.py'), '--method', method,
                   '--device', device, '--data-root', str(args.data_root), '--output', str(args.output)]
        for name in ('hidden_size', 'layers', 'epochs', 'batch_size', 'seed'):
            command.extend(['--'+name.replace('_', '-'), str(getattr(args, name))])
        if args.smoke:
            command.append('--smoke')
        print(command, flush=True)
        if not args.dry_run:
            result = subprocess.run(command, env=env)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
