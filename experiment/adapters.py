"""Native learning rules; shared hidden widths, not identical neuron dynamics.

TP and SPSN import their original model classes. ESPP imports EchoSpike and
simple_out, retaining its classwise predictive/contrastive sampling and manual
gradient updates. SNNFF's recurrent layers are extracted here from its original
top-level training script, with explicit sequence state instead of globals.
"""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked(loss):
    if not torch.isfinite(loss).all():
        raise FloatingPointError('Nonfinite native loss')
    return loss


class Adapter:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.optimizers, self.schedulers = [], []

    def parameters(self):
        return self.model.parameters()

    def checkpoint(self):
        return dict(model=self.model.state_dict(),
                    optimizers=[o.state_dict() for o in self.optimizers],
                    schedulers=[s.state_dict() for s in self.schedulers])

    def final_fit(self, loader):
        return None


class TP(Adapter):
    def __init__(self, config, device):
        super().__init__(config, device)
        sys.path.insert(0, str(ROOT / 'traces_propagation-main'))
        from models.tp_mlp import TP_MLP
        params = dict(l_vth=1., l_leak_m=.96, l_leak_t=.97, l_rst_type='soft',
                      l_out_leak_m=1., l_rec=False, norm_type='none',
                      surrogate_type='1', surrogate_scale=1., train_s=False)
        self.model = TP_MLP([config['hidden_size']] * config['layers'], config['inputs'],
                           config['classes'], config['batch_size'], params).to(device)
        self.optimizers = [torch.optim.Adam(p, lr=.0001) for _, p in self.model.get_layerwise_params()]
        self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=config['epochs'], eta_min=.000025)
                           for o in self.optimizers]

    def train_batch(self, x, y):
        self.model.train()
        self.model.reset_potential()
        target = F.one_hot(y, self.config['classes']).float()
        for t in range(x.shape[1]):
            logits, _ = self.model(x[:, t], target)
            if t == x.shape[1] - 1:
                loss = checked(F.cross_entropy(logits, y))
            for opt in self.optimizers:
                opt.zero_grad()
            self.model.update()
            for opt in self.optimizers:
                opt.step()
            self.model.detach_membrane_states()
        return float(loss.detach()), int((logits.detach().argmax(1) == y).sum())

    def predict(self, x):
        self.model.eval()
        self.model.reset_potential()
        for t in range(x.shape[1]):
            logits = self.model.inference(x[:, t])
        return logits


class SPSN(Adapter):
    def __init__(self, config, device):
        super().__init__(config, device)
        sys.path.insert(0, str(ROOT / 'Stochastic-Parallelizable-Spiking-Neuron-SPSN-main'))
        from network import create_network
        self.model = create_network(dict(neuron='SPSN-SB', nb_layers=config['layers'],
            input_size=config['inputs'], hidden_size=config['hidden_size'], nb_class=config['classes'],
            tau_mem=.02, tau_syn=.02, time_step=.001), device, config['steps']).to(device)
        self.optimizers = [torch.optim.Adamax(self.model.parameters(), lr=.001)]

    def train_batch(self, x, y):
        self.model.train()
        logits = self.model(x).mean(1)
        loss = checked(F.cross_entropy(logits, y))
        self.optimizers[0].zero_grad()
        loss.backward()
        self.optimizers[0].step()
        return float(loss.detach()), int((logits.detach().argmax(1) == y).sum())

    def predict(self, x):
        self.model.eval()
        return self.model(x).mean(1)


class FFLayer(nn.Module):
    def __init__(self, inputs, hidden, normalize=False):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate
        self.fc = nn.Linear(inputs, hidden)
        self.lif = snn.RLeaky(beta=.9, threshold=5., spike_grad=surrogate.atan(alpha=2),
                             linear_features=hidden, learn_threshold=True, learn_beta=True)
        self.normalize = normalize

    def forward(self, sequence):
        spike, mem = self.lif.init_rleaky()
        outputs = []
        for t in range(sequence.shape[1]):
            current = self.fc(sequence[:, t])
            if self.normalize:
                current = F.normalize(current)
            spike, mem = self.lif(current, spike, mem)
            outputs.append(spike)
        return torch.stack(outputs, 1)


class SNNFF(Adapter):
    def __init__(self, config, device):
        super().__init__(config, device)
        widths = [config['inputs'] + config['classes']] + [config['hidden_size']] * config['layers']
        self.model = nn.ModuleList([FFLayer(a, b, i > 0) for i, (a, b) in enumerate(zip(widths, widths[1:]))]).to(device)
        self.optimizers = [torch.optim.Adam(self.model.parameters(), lr=.001)]

    def encode(self, x, y):
        labels = x.new_zeros((len(x), x.shape[1], self.config['classes']))
        labels[torch.arange(len(x), device=x.device), :, y] = x.max()
        return torch.cat((labels, x), -1)

    @staticmethod
    def goodness(sequence):
        return sequence.sum(1).square().mean(1)

    @torch.no_grad()
    def predict(self, x):
        scores = []
        for label in range(self.config['classes']):
            h = self.encode(x, torch.full((len(x),), label, device=x.device, dtype=torch.long))
            goodness = 0
            for layer in self.model:
                h = layer(h)
                goodness = goodness + self.goodness(h)
            scores.append(goodness)
        return torch.stack(scores, 1)

    def train_batch(self, x, y):
        self.model.train()
        scores = self.predict(x)
        correct = int((scores.argmax(1) == y).sum())
        scores[torch.arange(len(x), device=x.device), y] = 0
        negative = torch.multinomial(scores.sqrt() + 1e-12, 1).squeeze(1)
        positive, negative = self.encode(x, y), self.encode(x, negative)
        losses = []
        for layer in self.model:
            pos, neg = layer(positive), layer(negative)
            loss = checked(F.silu(-4 * (self.goodness(pos) - self.goodness(neg))).mean())
            self.optimizers[0].zero_grad()
            loss.backward()
            self.optimizers[0].step()
            losses.append(float(loss.detach()))
            with torch.no_grad():
                positive, negative = layer(positive), layer(negative)
        return sum(losses) / len(losses), correct


class ESPP(Adapter):
    def __init__(self, config, device):
        super().__init__(config, device)
        native = module_at('native_espp_model', 'ESPP-main/model.py')
        self.native_data = module_at('native_espp_data', 'ESPP-main/data.py')
        beta = [.94, .96, .98]
        self.beta = [beta[min(i, len(beta)-1)] for i in range(config['layers'])]
        self.model = native.EchoSpike(config['inputs'], [config['hidden_size']] * config['layers'],
            c_y=[1.5, -1.5], beta=self.beta, device=str(device), recurrency_type='dense',
            n_time_steps=config['steps'], online=True, inp_thr=.05).to(device)
        self.model.reset(0)
        # Native readouts use manual gradients and CPU AdamW.
        self.readouts = nn.ModuleList([native.simple_out(n, config['classes'], beta=1.)
            for n in [config['inputs']] + [config['hidden_size']] * config['layers']])
        self.optimizers = [torch.optim.SGD([{'params': layer.fc.parameters(), 'lr': .0001}
                                          for layer in self.model.layers])]
        self.readout_optimizers = [torch.optim.AdamW(p.parameters(), lr=.001, weight_decay=0.)
                                   for p in self.readouts]
        self.optimizers += self.readout_optimizers
        self.bf, self.items = 0, 0
        self.target = None

    def parameters(self):
        return iter([*self.model.parameters(), *self.readouts.parameters()])

    def checkpoint(self):
        result = super().checkpoint()
        result['readouts'] = self.readouts.state_dict()
        return result

    def prepare_sampler(self, dataset):
        targets = [dataset[i][1] for i in range(len(dataset))]
        self.sampler = self.native_data.classwise_loader(dataset, targets, self.config['classes'],
                                                        self.config['batch_size'], seed=self.config['seed'])
        if any(len(pool) == 0 for pool in self.sampler.target_indeces):
            raise ValueError('ESPP requires at least one training sample for every class')
        self.target = [torch.randint(self.config['classes'], (1,)).item() for _ in range(self.config['batch_size'])]

    @torch.no_grad()
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = batches = 0
        # Preserve native sample-count boundaries and alternating bf across epochs.
        while self.items < epoch * len(self.sampler):
            data, self.target = self.sampler.next_item(self.target, contrastive=self.bf == -1)
            data = data.float().to(self.device)
            losses = []
            for step in data:
                _, _, loss, _ = self.model(step, torch.tensor(self.bf, device=self.device),
                                           inp_activity=step.mean(-1))
                checked(loss)
                losses.append(loss.mean())
                self.optimizers[0].step()
                self.optimizers[0].zero_grad()
            self.model.reset(self.bf)
            self.bf = 1 if self.bf != 1 else -1
            total_loss += float(torch.stack(losses).mean())
            batches += 1
            self.items += self.config['batch_size']
        return total_loss / batches, None

    @torch.no_grad()
    def final_fit(self, loader):
        self.model.eval()
        samples = [[] for _ in self.readouts]
        labels = []
        for x, y in loader:
            x = x.to(self.device)
            self.model.reset(0)
            # Native get_samples stores hidden spike counts as uint8.
            counts = [torch.zeros((len(x), self.config['hidden_size']), dtype=torch.uint8)
                      for _ in self.model.layers]
            for t in range(x.shape[1]):
                spikes, _, _, _ = self.model(x[:, t], 0)
                for index, spike in enumerate(spikes):
                    counts[index] += spike.cpu().to(torch.uint8)
            for buffer, value in zip(samples, [x.sum(1).cpu(), *counts]):
                buffer.append(value)
            labels.append(y.cpu())
        samples = [torch.cat(values) for values in samples]
        labels = torch.cat(labels)
        order = np.arange(len(labels))
        np.random.shuffle(order)
        total_loss = 0.
        for start in range(0, len(labels), self.config['batch_size']):
            ids = order[start:start + self.config['batch_size']]
            for projection, opt, values in zip(self.readouts, self.readout_optimizers, samples):
                projection.train()
                projection.reset()
                _, logits = projection(values[ids].float(), labels[ids])
                loss = checked(F.cross_entropy(logits, labels[ids]))
                total_loss += float(loss) * len(ids) / len(self.readouts)
                projection.reset()
                opt.step()
                opt.zero_grad()
        return total_loss / len(labels)

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        self.model.reset(0)
        projection = self.readouts[-1]
        projection.eval()
        projection.reset()
        for t in range(x.shape[1]):
            spikes, _, _, _ = self.model(x[:, t], 0)
            _, logits = projection(spikes[-1].cpu(), None)
        return logits.to(self.device)


ADAPTERS = dict(tp=TP, espp=ESPP, snnff=SNNFF, spsn=SPSN)
