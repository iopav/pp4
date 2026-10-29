"""Native training with training-only energy windows and one final test."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data import datasets
from adapters import ADAPTERS, ESPP
from bp_memory import SavedTensorMemory, BP_MEMORY_FIELD, BP_MEMORY_VERSION
from state_memory import optimizer_state_bytes, AUX_MEMORY_VERSION
from completion import skip_completed
from energy import TotalEnergyCounter
from startup_power import report_startup_power


def allocate_block_energy(rows, joules, total_energy, total_seconds):
    duration = sum(row['epoch_seconds'] for row in rows)
    if duration <= 0:
        raise ValueError('Nonpositive training duration')
    assigned = 0.
    for i, row in enumerate(rows):
        seconds = row['epoch_seconds']
        energy = None if joules is None else (joules-assigned if i == len(rows)-1 else joules*seconds/duration)
        if energy is not None:
            assigned += energy
        total_seconds += seconds
        total_energy = total_energy+energy if total_energy is not None and energy is not None else None
        row.update(energy_j=energy, power_w=None if energy is None else energy/seconds,
                   total_energy_j=total_energy, total_power_w=None if total_energy is None else total_energy/total_seconds,
                   total_seconds=total_seconds)
    return total_energy, total_seconds


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def metric_row(adapter, epoch, loss, accuracy, peak, seconds):
    state_bytes = optimizer_state_bytes(SimpleNamespace(state=[o.state for o in adapter.optimizers]))
    return dict(epoch=epoch, para_mem_bytes=sum(p.numel()*p.element_size() for p in adapter.parameters()),
                optimizer_state_bytes=state_bytes, auxiliary_mem_bytes=state_bytes+peak,
                **{BP_MEMORY_FIELD: peak}, train_loss=loss, train_acc=accuracy, final_test_acc=None,
                energy_j=None, power_w=None, total_energy_j=None, total_power_w=None,
                epoch_seconds=seconds, total_seconds=None)


@torch.no_grad()
def evaluate(adapter, loader, device):
    correct = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = adapter.predict(x)
        if logits.shape != (len(y), adapter.config['classes']) or not torch.isfinite(logits).all():
            raise FloatingPointError('Invalid classification output')
        correct += int((logits.argmax(1) == y).sum())
        count += len(y)
    return correct/count


def train_loop(adapter, train_loader, train_eval_loader, test_loader, config, destination, counter):
    device = adapter.device
    rows, pending = [], []
    total_energy = total_seconds = 0.
    initial = None

    def flush():
        nonlocal total_energy, total_seconds
        _, final = counter.read()
        joules = (final-initial)/1000 if initial is not None and final is not None else None
        total_energy, total_seconds = allocate_block_energy(pending, joules, total_energy, total_seconds)
        rows.extend(pending)
        pending.clear()

    def save():
        with (destination/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    if isinstance(adapter, ESPP):
        adapter.prepare_sampler(train_loader.dataset)
    for epoch in range(1, config['epochs']+1):
        if not pending:
            _, initial = counter.read()
        sync(device)
        started = time.perf_counter()
        peak = 0
        if isinstance(adapter, ESPP):
            loss, accuracy = adapter.train_epoch(epoch)
        else:
            total_loss = correct = count = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                with SavedTensorMemory(adapter.parameters(), device) as memory:
                    loss, batch_correct = adapter.train_batch(x, y)
                peak = max(peak, memory.peak_bytes)
                total_loss += loss*len(y)
                correct += batch_correct
                count += len(y)
            loss, accuracy = total_loss/count, correct/count
        for scheduler in adapter.schedulers:
            scheduler.step()
        sync(device)
        pending.append(metric_row(adapter, epoch, loss, accuracy, peak, time.perf_counter()-started))
        if len(pending) == 5 or epoch == config['epochs']:
            block_size = len(pending)
            flush()
            save()
            for row in rows[-block_size:]:
                print(json.dumps(row), flush=True)
    if isinstance(adapter, ESPP):
        _, initial = counter.read()
        sync(device)
        started = time.perf_counter()
        loss = adapter.final_fit(train_eval_loader)
        sync(device)
        pending.append(metric_row(adapter, config['epochs'], loss, None, 0, time.perf_counter()-started))
        flush()
        rows[-1]['train_acc'] = evaluate(adapter, train_eval_loader, device)
    rows[-1]['final_test_acc'] = evaluate(adapter, test_loader, device)
    save()
    print('[final] ' + json.dumps(rows[-1]), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=ADAPTERS, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hidden-size', type=int, default=49)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.hidden_size, args.layers, args.epochs, args.batch_size) < 1:
        parser.error('architecture and training dimensions must be positive')
    report_startup_power(args.device)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    train, test = datasets(args.data_root, False, 'texture-mini')
    config = dict(method=args.method, dataset='texture-mini', data_root=str(args.data_root.resolve()),
        hidden_size=args.hidden_size, layers=args.layers, epochs=1 if args.smoke else args.epochs,
        batch_size=args.batch_size, seed=args.seed, device=str(device), inputs=train.inputs,
        classes=train.classes, steps=min(16, train.steps) if args.smoke else train.steps, smoke=args.smoke,
        protocol='pp4_texture_native_v2_online_gradient_handoff', energy_window_epochs=5,
        energy_measurement_version='nvml_total_train_block_time_allocation_v1',
        bp_memory_version=BP_MEMORY_VERSION, auxiliary_memory_version=AUX_MEMORY_VERSION,
        accuracy_reporting_version='online_except_espp_final_readout_v1',
        loss_definition={'tp':'last_step_cross_entropy', 'spsn':'mean_time_cross_entropy',
                         'snnff':'mean_layer_native_swish', 'espp':'mean_layer_echo_loss_final_row_readout_ce'}[args.method])
    if args.smoke:
        def small(dataset):
            dataset.steps = config['steps']
            indices = [i for label in range(dataset.classes)
                       for i in np.flatnonzero(dataset.y == label)[:2]]
            return Subset(dataset, indices)
        train, test = small(train), small(test)
    config.update(train_samples=len(train), test_samples=len(test))
    destination = args.output / 'texture-mini' / ('smoke' if args.smoke else 'full') / args.method / f'seed_{args.seed}'
    adapter = ADAPTERS[args.method](config, device)
    config['optimizers'] = [dict(name=type(opt).__name__, defaults=opt.defaults) for opt in adapter.optimizers]
    config['schedulers'] = [dict(name=type(s).__name__, state=s.state_dict()) for s in adapter.schedulers]
    if args.method == 'espp':
        config['hidden_beta'] = adapter.beta
        config['readout_epochs'] = 1
    if skip_completed(destination, config):
        return
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    counter = TotalEnergyCounter(device)
    try:
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed))
        ordered = DataLoader(train, batch_size=args.batch_size)
        test_loader = DataLoader(test, batch_size=args.batch_size)
        train_loop(adapter, train_loader, ordered, test_loader, config, destination, counter)
        for p in adapter.parameters():
            if not torch.isfinite(p).all():
                raise FloatingPointError('Nonfinite saved parameters')
        torch.save(dict(**adapter.checkpoint(), config=config, completed_epochs=config['epochs']), destination/'model.pt')
        (destination/'status.json').write_text(json.dumps(dict(status='success', epochs=config['epochs'])))
    except BaseException:
        (destination/'status.json').write_text(json.dumps(dict(status='failed', error=traceback.format_exc())))
        raise
    finally:
        counter.close()


if __name__ == '__main__':
    main()
