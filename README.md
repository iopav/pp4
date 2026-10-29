# PP4 Texture-mini native-method comparison

The supported entry point is `run_experiments.py`. It runs TP, ESPP, SNNFF and
SPSN sequentially with one hidden layer of 49 units. Edit `BACKBONE` in that
entry point or pass `--hidden-size` and `--layers`; these controls reach every
model. Output layers are excluded from the hidden-layer count. This is a common
architecture size, not a claim of identical neuron dynamics or parameter counts.

```bash
pip install -r requirements-pp4.txt
python run_experiments.py --data-root /path/to/texture-mini
python run_experiments.py --hidden-size 64 --layers 2 --methods tp espp snnff spsn
python run_experiments.py --cpu --smoke --batch-size 8 --output outputs/smoke_check
sbatch submit_experiments.sbatch --data-root /path/to/texture-mini
python -m unittest discover -s experiment -p 'test_*.py'
```

Defaults: 100 epochs, batch size 128, seed 57, GPU 0 (or the first inherited
allocation), and `data/texture-mini/texture_spikesT450_T77.npz` inside this
repository. Supply `--data-root` to use another directory or the NPZ itself.
The file supplies the original training/test splits: 960/240 examples, 77 time
steps, 9 input channels and 12 classes. Dimensions are obtained from the loader;
inputs are not re-encoded. Smoke runs retain two examples per class and the
first 16 steps, use one epoch, and preserve the requested model size. They do
not establish convergence. `ffsnn` is accepted as an alias for `snnff`.

## Native learning rules

| Method | Training and classifier | Optimizer |
| --- | --- | --- |
| TP | Original TP_MLP, local update each time step; final-step classifier | Layerwise Adam 0.0001, native cosine schedule |
| ESPP | Original EchoSpike, online predictive/contrastive classwise sampling; final native simple_out readout | Backbone SGD 0.0001; readouts AdamW 0.001, weight decay 0 |
| SNNFF | Recurrent snnTorch layers, native swish goodness loss and hard-label sampling | Adam 0.001 |
| SPSN | Original SPSN-SB, mean temporal readout | Adamax 0.001 |

TP retains its target propagator as learning machinery, and its parameters are
included in retained parameter storage. ESPP beta values follow the native
profile (.94, .96, .98), truncated to the layer count or extended with .98.
ESPP trains the original input and hidden-layer readouts once after backbone
training, reports the last hidden layer's classifier, and accounts for all
retained readouts. Their native manual-gradient computation runs on CPU.
The upstream ESPP online branch returned a local gradient without assigning it
to the weight. PP4 connects that existing gradient to `fc.weight.grad` before
SGD, and tests that backbone weights actually change; the gradient formula and
optimizer are unchanged.
SNNFF reserves a separate prefix of 12 label channels, preserving all 9 sensor
channels. Its 12 output scores are candidate-label goodness values; no linear
classification head is added. Its recurrent layers were extracted from
`SNN_FF-main/FF_SNN/SNNFF_SHD.py`; explicit sequence state replaces global
spike buffers so arbitrary positive layer counts work. Upstream reference
scripts are not the PP4 execution interface.

## Metrics and boundaries

`outputs/texture-mini/{full,smoke}/METHOD/seed_SEED/` contains `config.json`,
`metrics.csv`, `model.pt` and `status.json`. Configurations include architecture,
optimizer defaults and metric versions. Matching successful runs are skipped;
incomplete or different runs are preserved and require a new `--output`.
Checkpoints include every optimizer and scheduler state. Exact interrupted-run
resume is not implemented.

`para_mem_bytes` counts all retained model parameters by their actual dtype.
`optimizer_state_bytes` counts epoch-end optimizer storage across all optimizers,
deduplicated by underlying storage. `hiddenstatememfor bp` is the maximum live
nonparameter autograd-saved storage over training batches on the training device.
`auxiliary_mem_bytes` is their sum, not peak runtime allocation; manual local
learning traces, gradients and other working buffers are outside that metric.
ESPP uses manual gradients without autograd and therefore has zero BP storage.

Five consecutive epochs form an energy window with no reporting evaluation or
CSV/log writes inside. `energy_j` allocates the NVML cumulative energy difference
by epoch training duration; `power_w = energy_j / epoch_seconds`. Total power is
total energy divided by total measured training time. The ESPP final readout fit
has its own window and final CSV row, reusing the last epoch number. This row's
loss is readout cross entropy rather than backbone echo loss. Missing telemetry
is blank. Equal counter readings remain zero; very short windows can be below
sensor update resolution. GPU board energy does not measure CPU readout energy.

TP/SPSN/SNNFF report sample-weighted online training accuracy; SNNFF reuses the
scores needed to sample negative labels. ESPP has no class prediction while
training its backbone, so those train accuracy cells are blank. After the final
readout fit it evaluates the training split outside the energy window. Every
method evaluates the test split exactly once, after all fitting, and only the
last CSV row contains `final_test_acc`. No test-based checkpoint selection occurs.
Loss meanings are method-specific and recorded in `config.json`.

The launcher samples pre-training GPU board power for approximately three
seconds at 0.1-second intervals and prints one mean. Child methods do not repeat
it; standalone `experiment/train.py` invocations sample once themselves. This is
not guaranteed idle power and is not subtracted from training energy. CPU runs
and unavailable telemetry print an explicit unavailable reason.

## Validation limits

CPU and CUDA smoke runs use real Texture-mini data; unit tests cover alternative
architecture sizes, native optimizer types and parameter updates, isolated label
encoding, final-only tests, five-epoch energy allocation and ESPP readout timing.
Full 100-epoch convergence and Slurm execution are not validated by these checks.
The implementation is self-contained and does not import files from pp3.
