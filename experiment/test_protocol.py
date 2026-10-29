"""Structural, learning and accounting regressions; CPU, tiny synthetic data."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import torch
from torch.utils.data import DataLoader, TensorDataset
from adapters import ADAPTERS, ESPP
from train import train_loop, metric_row


def config(method, hidden=49, layers=1, epochs=1):
    return dict(method=method, inputs=9, classes=12, steps=8, hidden_size=hidden,
                layers=layers, epochs=epochs, batch_size=4, seed=57)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(57)
        torch.set_num_threads(2)

    def test_shapes_and_configurable_hidden_layers(self):
        for name in ADAPTERS:
            for width, layers in ((49, 1), (7, 2)):
                with self.subTest(method=name, width=width, layers=layers):
                    adapter = ADAPTERS[name](config(name, width, layers), torch.device('cpu'))
                    x = torch.randint(0, 2, (4, 8, 9)).float()
                    with torch.no_grad():
                        out = adapter.predict(x)
                    self.assertEqual(out.shape, (4, 12))
                    self.assertTrue(torch.isfinite(out).all())
                    if name == 'tp':
                        self.assertEqual(adapter.model.hidden_layers, [width]*layers)
                    elif name == 'spsn':
                        self.assertEqual(len(adapter.model), layers+1)
                        self.assertTrue(all(m.hidden_size == width for m in list(adapter.model)[:-1]))
                    elif name == 'snnff':
                        self.assertEqual(len(adapter.model), layers)
                        self.assertTrue(all(m.fc.out_features == width for m in adapter.model))
                        encoded = adapter.encode(x, torch.arange(4))
                        torch.testing.assert_close(encoded[:, :, 12:], x)
                        self.assertEqual(encoded.shape[-1], 21)
                    else:
                        self.assertEqual(adapter.model.num_hidden, [width]*layers)
                        self.assertEqual(len(adapter.beta), layers)

    def test_updates_native_optimizers_and_state_accounting(self):
        expected = dict(tp=['Adam']*2, snnff=['Adam'], spsn=['Adamax'], espp=['SGD','AdamW','AdamW'])
        for name in ADAPTERS:
            with self.subTest(method=name):
                adapter = ADAPTERS[name](config(name), torch.device('cpu'))
                self.assertEqual([type(o).__name__ for o in adapter.optimizers], expected[name])
                before = [p.detach().clone() for p in adapter.parameters()]
                x = torch.randint(0, 2, (24, 8, 9)).float()
                y = torch.arange(24) % 12
                if isinstance(adapter, ESPP):
                    dataset = TensorDataset(x, y)
                    adapter.prepare_sampler(dataset)
                    adapter.train_epoch(1)
                    self.assertTrue(any(not torch.equal(a, b) for a, b in
                                        zip(before, adapter.model.parameters())),
                                    'ESPP online local gradients must reach backbone SGD')
                    adapter.final_fit(DataLoader(dataset, batch_size=4))
                else:
                    # SNNFF's high threshold can produce no spikes on tiny binary
                    # sequences. Exercise a spiking input for the update check.
                    adapter.train_batch(x[:4] * (5 if name == 'snnff' else 1), y[:4])
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, adapter.parameters())))
                row = metric_row(adapter, 1, 0., None, 123, 1.)
                self.assertGreater(row['optimizer_state_bytes'], 0)
                self.assertEqual(row['auxiliary_mem_bytes'], row['optimizer_state_bytes']+123)

    def test_five_epoch_windows_final_test_once(self):
        c = config('spsn', hidden=7, epochs=7)
        adapter = ADAPTERS['spsn'](c, torch.device('cpu'))
        data = DataLoader(TensorDataset(torch.ones(4, 8, 9), torch.arange(4)), batch_size=4)
        counter = Mock()
        counter.read.side_effect = [(0,0), (1,5000), (2,5000), (3,7000)]
        def evaluate(*args):
            self.assertEqual(counter.read.call_count, 4)
            return .25
        with tempfile.TemporaryDirectory() as root, patch('train.evaluate', side_effect=evaluate) as evaluation, \
             contextlib.redirect_stdout(io.StringIO()):
            rows = train_loop(adapter, data, data, data, c, Path(root), counter)
        evaluation.assert_called_once()
        self.assertEqual(len(rows), 7)
        self.assertAlmostEqual(sum(r['energy_j'] for r in rows), 7.)
        self.assertTrue(all(r['final_test_acc'] is None for r in rows[:-1]))
        self.assertEqual(rows[-1]['final_test_acc'], .25)

    def test_espp_readout_window_and_final_evaluations(self):
        c = config('espp', hidden=7, epochs=2)
        adapter = ESPP(c, torch.device('cpu'))
        data = DataLoader(TensorDataset(torch.randint(0,2,(24,8,9)).float(), torch.arange(24)%12), batch_size=4)
        counter = Mock()
        counter.read.side_effect = [(0,0), (1,5000), (2,5000), (3,7000)]
        with tempfile.TemporaryDirectory() as root, patch('train.evaluate', return_value=.25) as evaluation, \
             contextlib.redirect_stdout(io.StringIO()):
            rows = train_loop(adapter, data, data, data, c, Path(root), counter)
        self.assertEqual(evaluation.call_count, 2)  # final train and final test
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r['train_acc'] is None for r in rows[:-1]))
        self.assertEqual(rows[-1]['energy_j'], 2.)
        self.assertEqual(rows[-1]['total_energy_j'], 7.)


if __name__ == '__main__':
    unittest.main()
