# Grokking pipeline

A modular PyTorch pipeline to reproduce and extend Nanda et al. (2023)
[*Progress measures for grokking via mechanistic interpretability*](https://arxiv.org/abs/2301.05217)
on modular addition mod 113.

The pipeline supports plug-and-play optimizers (AdamW, Muon, SOAP, SGD),
per-layer optimizer assignment (hybrid setups), multi-seed runs with
median + min/max bands, and resume-tolerant grid search.

---

## Table of contents

- [Project structure](#project-structure)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Architecture overview](#architecture-overview)
- [Building OptimizerSpecs](#building-optimizerspecs)
- [Per-layer optimizer modularity (`ParamFilter`)](#per-layer-optimizer-modularity-paramfilter)
- [Launching a single-seed training](#launching-a-single-seed-training)
- [Launching a multi-seed training](#launching-a-multi-seed-training)
- [Resuming a training](#resuming-a-training)
- [Grid search](#grid-search)
- [What gets saved on disk](#what-gets-saved-on-disk)
- [The `history` dictionary](#the-history-dictionary)
- [Adding a new optimizer](#adding-a-new-optimizer)
- [Troubleshooting](#troubleshooting)

---

## Project structure

```
baseline/
├── model.py                # The transformer (Nanda architecture)
├── pipeline.py             # Engine: OptimizerSpec, Trainer, grid_search
├── plots.py                # Visualization: plot_curves, plot_seeds_overlay, ...
├── optimizer/              # Vendored third-party optimizers
│   ├── __init__.py
│   ├── muon.py             # Keller Jordan's Muon
│   └── soap.py             # Nikhil Vyas' SOAP
├── runs/                   # Auto-created. Holds all training artifacts.
│   ├── adamw_baseline/
│   │   └── seed0/ ...
│   └── grid/
│       └── adamw/
│           └── lr0.001__weight_decay1/seed0/ ...
├── multiseed_baseline.ipynb     # Baseline: AdamW × 5 seeds
├── grid_search.ipynb            # AdamW grid search (Nanda methodology)
├── grid_search_muon_soap.ipynb  # Muon + SOAP grid searches
└── README.md
```

---

## Setup

### Python + dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib einops
```

Tested on Python 3.11.

### Device support

The pipeline auto-detects the best device in `model.py:Config`:

| Priority | Device | Notes |
|---|---|---|
| 1 | `cuda` | Recommended for full speed |
| 2 | `mps` | Apple Silicon. Cross-entropy falls back to float32 (no float64 on MPS) |
| 3 | `cpu` | Slow but works for tiny smoke tests |

No manual device flag needed — it just works.

---

## Quick start

A complete single-seed AdamW training in 10 lines:

```python
import numpy as np
import torch as t
from model import Config
from pipeline import OptimizerSpec, Trainer
from plots import plot_curves

t.manual_seed(0); np.random.seed(0)

config = Config(p=113, frac_train=0.3, num_epochs=25_000)

specs = [
    OptimizerSpec('adamw', lr=1e-3, weight_decay=1.0,
                  extra={'betas': (0.9, 0.98)}),
]

trainer = Trainer(config, specs, seed=0, label='quickstart',
                  eval_every=50, verbose_every=2000)
trainer.fit()
trainer.save_run('runs/quickstart/seed0')
plot_curves(trainer.history, title="Quick start")
```

Expected: `test_acc → 1.0` somewhere around epoch ~10 000.

---

## Architecture overview

The pipeline has 3 layers:

```
┌────────────────────────────────────────────────────────────────┐
│  USER LAYER                                                     │
│  - You write OptimizerSpec(...) objects (declarative)           │
│  - You call Trainer(...).fit() or run_multi_seed(...) or        │
│    grid_search(...) (imperative entry points)                   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  PIPELINE LAYER (pipeline.py)                                   │
│  - OptimizerSpec dataclass  (frozen, hashable)                  │
│  - _OPTIMIZER_REGISTRY      (name -> builder function)          │
│  - build_optimizers()       (partition params via ParamFilters) │
│  - Trainer class            (fit / step / save_run / from_run)  │
│  - run_multi_seed()         (1 trainer per seed)                │
│  - grid_search()            (cartesian product × seeds)         │
│  - aggregate_grid()         (analysis table)                    │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  THIRD-PARTY LAYER                                              │
│  - model.py (your Transformer + Config)                         │
│  - optimizer/muon.py, optimizer/soap.py (vendored)              │
│  - torch.optim.AdamW, torch.optim.SGD                           │
└────────────────────────────────────────────────────────────────┘
```

**Key design principle**: the pipeline never knows which optimizer class
is being built. It only knows how to dispatch by string name to a builder
function. Adding a new optimizer = 3 lines (see [Adding a new optimizer](#adding-a-new-optimizer)).

---

## Building OptimizerSpecs

An `OptimizerSpec` is a **frozen dataclass** that fully describes one optimizer
applied to a subset of model parameters.

```python
@dataclass(frozen=True)
class OptimizerSpec:
    name:         str               # 'adamw' | 'muon' | 'sgd' | 'soap'
    lr:           float
    weight_decay: float = 0.0
    param_filter: Optional[Callable[[str, Tensor], bool]] = None
    extra:        dict = {}         # algorithm-specific kwargs
```

### One spec per optimizer flavor

```python
# AdamW (Nanda baseline)
OptimizerSpec('adamw', lr=1e-3, weight_decay=1.0,
              extra={'betas': (0.9, 0.98)})

# Muon (Keller Jordan)
OptimizerSpec('muon', lr=5e-3, weight_decay=0.5,
              extra={'momentum': 0.95})

# SOAP (Nikhil Vyas)
OptimizerSpec('soap', lr=3e-3, weight_decay=0.01,
              extra={'betas': (0.95, 0.95), 'precondition_frequency': 10})

# SGD
OptimizerSpec('sgd', lr=1e-2, weight_decay=0.5,
              extra={'momentum': 0.9, 'nesterov': True})
```

**Anything that is not `lr` / `weight_decay` goes into `extra`.** The builder
function unpacks `extra` as kwargs, so any keyword the underlying optimizer
accepts will work.

---

## Per-layer optimizer modularity (`ParamFilter`)

Some optimizers (Muon, SOAP) should only be applied to **2-D internal matrices**.
Embeddings, biases, and the unembed layer should use AdamW.

The pipeline supports this by accepting a `param_filter` predicate on each spec.

### Signature

```python
ParamFilter = Callable[[str, Tensor], bool]
#                       ^^^^   ^^^^^^
#                       name   the actual parameter tensor
```

You need **both** because:
- `name` carries **structural** info (`'attn'`, `'mlp'`, `'embed'`)
- `tensor` carries **shape** info (`p.ndim`, `p.shape`, `p.dtype`)

### The canonical Muon / SOAP filter

```python
FILTER_INTERNAL_2D = lambda n, p: p.ndim >= 2 and 'embed' not in n
```

Applied to our model:

| Param name | ndim | matches? | Goes to |
|---|---|---|---|
| `embed.W_E` | 2 | ❌ (contains 'embed') | catch-all → AdamW |
| `pos_embed.W_pos` | 2 | ❌ ('embed' substring) | catch-all → AdamW |
| `blocks.0.attn.W_K` | 3 | ✅ | Muon / SOAP |
| `blocks.0.mlp.W_in` | 2 | ✅ | Muon / SOAP |
| `blocks.0.mlp.b_in` | 1 | ❌ (ndim < 2) | catch-all → AdamW |
| `unembed.W_U` | 2 | ❌ ('embed' substring) | catch-all → AdamW |

### How specs interact

When you pass a list of specs:

```python
specs = [
    OptimizerSpec('muon', ..., param_filter=FILTER_INTERNAL_2D),
    OptimizerSpec('adamw', ..., param_filter=None),   # catch-all
]
```

`build_optimizers` does this:
1. Sorts: catch-all specs (where `param_filter is None`) are moved to the end.
2. For each parameter, walks the specs in order. **First match wins**.
3. If no spec claims a parameter, raises an error.

So if you pass `[Muon_with_filter, AdamW_catchall]`:
- Internal 2-D matrices get Muon.
- Everything else (embeddings, biases, unembed) gets AdamW.

### Inspecting the partition

Use `verbose_build=True` to see which params each spec captured:

```python
trainer = Trainer(config, specs, seed=0, verbose_build=True)
# Spec order after sorting:
#   0 : muon(filtered, lr=0.005, wd=0.5, momentum=0.95)
#   1 : adamw(ALL, lr=0.001, wd=1.0, betas=(0.9, 0.98))
# muon(filtered, ...): 6 tensors, 197,632 params
#       |__ blocks.0.attn.W_K
#       |__ blocks.0.attn.W_Q
#       |__ ...
# adamw(ALL, ...): 5 tensors, 29,184 params
#       |__ embed.W_E
#       |__ ...
```

---

## Launching a single-seed training

```python
from model import Config
from pipeline import OptimizerSpec, Trainer

config = Config(p=113, frac_train=0.3, num_epochs=25_000)

specs = [
    OptimizerSpec('adamw', lr=1e-3, weight_decay=1.0,
                  extra={'betas': (0.9, 0.98)}),
]

trainer = Trainer(
    config, specs,
    seed=0,
    label='nanda_baseline',
    eval_every=50,           # log eval metrics every N steps
    fourier_every=None,      # disable Fourier metrics (not wired yet)
    warmup_steps=10,
    verbose_every=2000,      # print progress every N steps
    verbose_build=True,
)

history = trainer.fit()       # blocking, runs until num_epochs

trainer.save_run('runs/nanda_baseline/seed0')
```

### What `trainer.fit()` returns

A `history` dict with all logged metrics (see [The `history` dictionary](#the-history-dictionary)).

### Continuing a run

`fit(num_epochs=...)` resumes from `self.epoch`:

```python
trainer.fit(num_epochs=5000)    # train to 5k
# inspect, plot, ...
trainer.fit(num_epochs=40000)   # continue to 40k
```

---

## Launching a multi-seed training

```python
from pipeline import OptimizerSpec, run_multi_seed

specs = [
    OptimizerSpec('adamw', lr=1e-3, weight_decay=1.0,
                  extra={'betas': (0.9, 0.98)}),
]

results = run_multi_seed(
    config, specs,
    seeds=[0, 1, 2, 3, 4],
    label_prefix='adamw_baseline',
    save_root='runs/adamw_baseline',   # creates seed0/, seed1/, ...
    eval_every=50,
    warmup_steps=10,
    verbose_every=2000,
)
# results: dict[int, Trainer]
#   {0: trainer0, 1: trainer1, ...}
```

### Visualizing multi-seed runs

```python
from plots import plot_seeds_overlay, plot_seeds_band, plot_seeds_l2_norm

plot_seeds_overlay(results, title="AdamW × 5 seeds", aggregate='median')
# 3 rows × 2 cols layout:
#   row 1: train+test loss (lin x | log x)
#   row 2: train+test accuracy (lin x | log x)
#   row 3: ||W||^2 (lin x | log x)
```

Aggregation options:
- `'median'` — median curve across seeds (recommended)
- `'mean'` — mean curve
- `'min_max'` — median + min/max envelope

---

## Resuming a training

Every `save_run(folder)` produces a self-contained folder. Reload with:

```python
from pipeline import Trainer

trainer = Trainer.from_run('runs/nanda_baseline/seed0')
trainer.fit(num_epochs=40000)   # continue from where it stopped
```

`from_run` reconstructs:
- `Config` from `meta.json`
- `OptimizerSpec`s from `meta.json`
- Model weights from `model.pt`
- Optimizer + scheduler states from `optim.pt`
- `history` dict from `history.json`
- The `epoch` counter

### ⚠️ Caveat for hybrid setups with `param_filter`

`param_filter` is a Python lambda → not serializable to JSON.
For runs using hybrid specs (Muon / SOAP), pass the specs explicitly:

```python
specs = [
    OptimizerSpec('muon', lr=5e-3, weight_decay=0.5,
                  param_filter=FILTER_INTERNAL_2D,
                  extra={'momentum': 0.95}),
    OptimizerSpec('adamw', lr=1e-3, weight_decay=1.0,
                  extra={'betas': (0.9, 0.98)}),
]
trainer = Trainer.from_run('runs/muon_hybrid/seed0', specs=specs)
```

---

## Grid search

`grid_search` is the **resume-tolerant** workhorse for hyperparameter sweeps.

### The `spec_builder` pattern

Instead of hardcoding which hyperparameters you can sweep, you pass a
**function** that produces specs from named kwargs:

```python
def make_adamw(lr, weight_decay):
    return [OptimizerSpec('adamw', lr=lr, weight_decay=weight_decay,
                          extra={'betas': (0.9, 0.98)})]
```

The keys of `param_grid` must match the kwargs of `spec_builder`:

```python
param_grid = {
    'lr':           [1e-3, 3e-3, 1e-2],   # log-spaced
    'weight_decay': [0.3, 1.0, 3.0],
}
```

### Launching the search

```python
from pipeline import grid_search, aggregate_grid, print_grid_table

results = grid_search(
    config,
    make_adamw,
    param_grid,
    seeds=[0, 1],                       # phase 1: 2 seeds for detection
    save_root='runs/grid/adamw',
    num_epochs=25_000,                  # overrides config.num_epochs
    eval_every=50,
    warmup_steps=10,
    verbose_every=5_000,
)
# results: dict[tuple, dict[int, Trainer]]
#   {(1e-3, 1.0): {0: t0, 1: t1}, (1e-3, 3.0): {...}, ...}
```

### The 2-phase methodology (recommended)

```python
# Phase 1: fan-out detection on 2 seeds × full grid
results_p1 = grid_search(config, spec_builder, PARAM_GRID,
                          seeds=[0, 1], save_root=SAVE_ROOT,
                          num_epochs=NUM_EPOCHS)

# Pick winner from aggregate
rows_p1 = aggregate_grid(results_p1, param_keys=list(PARAM_GRID.keys()))
print_grid_table(rows_p1, param_keys=list(PARAM_GRID.keys()))
best = rows_p1[0]   # sorted by epoch_grok_median ascending

# Phase 2: confirmation on 5 seeds, ONLY winner.
# Same save_root + same num_epochs -> seeds 0+1 are reloaded for free.
BEST_GRID = {'lr': [best['lr']], 'weight_decay': [best['weight_decay']]}
results_p2 = grid_search(config, spec_builder, BEST_GRID,
                          seeds=[0, 1, 2, 3, 4], save_root=SAVE_ROOT,
                          num_epochs=NUM_EPOCHS)
```

→ Phase 2 only retrains seeds 2, 3, 4 (the missing ones). Seeds 0+1 are
reloaded from disk via `Trainer.from_run`.

### Sweeping hybrid Muon / SOAP

```python
FILTER_INTERNAL_2D = lambda n, p: p.ndim >= 2 and 'embed' not in n

def make_muon_hybrid(lr_muon, wd_muon):
    return [
        OptimizerSpec('muon', lr=lr_muon, weight_decay=wd_muon,
                      param_filter=FILTER_INTERNAL_2D,
                      extra={'momentum': 0.95}),
        OptimizerSpec('adamw', lr=1e-3, weight_decay=1.0,
                      extra={'betas': (0.9, 0.98)}),
    ]

PARAM_GRID_MUON = {
    'lr_muon': [1e-3, 3e-3, 1e-2],
    'wd_muon': [0.3, 1.0, 3.0],
}

results_muon = grid_search(config, make_muon_hybrid, PARAM_GRID_MUON,
                            seeds=[0, 1], save_root='runs/grid/muon_hybrid',
                            num_epochs=25_000)
```

See `grid_search_muon_soap.ipynb` for the full Muon + SOAP example.

### Interpreting the result table

`aggregate_grid` returns rows sorted by:
1. `did_grok_ratio >= robustness_thresh` (filter, default 0.5)
2. `epoch_grok_median` ascending (primary key)

`print_grid_table` shows:

```
lr     | wd  | did_grok | mem_med | eg_med | gap_med | l2_final | final_acc
------ | --- | -------- | ------- | ------ | ------- | -------- | ---------
0.001  | 1.0 | 5/5      |  2 300  |  9 500 |  7 200  |    42.1  | 1.0000   ← winner
0.005  | 0.3 | 5/5      |  7 400  |  9 200 |  1 800  |   180.3  | 1.0000
0.001  | 0.3 | 0/5      |  3 100  |  None  |  None   |   312.4  | 0.0445   ← invalid
```

Columns:
- `did_grok` — fraction of seeds that crossed `test_acc > 0.99`
- `mem_med` — median epoch when `train_acc > 0.99` (memorization)
- `eg_med` — median epoch when `test_acc > 0.99` (grokking)
- `gap_med` — `eg_med - mem_med` (signature of grokking dynamics)
- `l2_final` — final `||W||^2` (proxy for circuit cleanliness)
- `final_acc` — final test accuracy

---

## What gets saved on disk

After a `trainer.save_run(folder)`:

```
folder/
├── meta.json        # ~1 KB    — Config + Specs + epoch + timestamp
├── history.json     # ~50 KB   — all logged metrics across training
├── model.pt         # ~900 KB  — model.state_dict()
└── optim.pt         # ~3 MB    — optimizer + scheduler state_dicts
                                  (needed for resume; ~4 MB total per seed)
```

### `meta.json` example

```json
{
  "label": "adamw_baseline",
  "seed": 0,
  "epoch": 25000,
  "saved_at": "2026-06-02T14:32:18",
  "config": {
    "p": 113, "d_model": 128, "d_mlp": 512, "num_heads": 4,
    "frac_train": 0.3, "num_epochs": 25000, "seed": 0, ...
  },
  "specs": [
    {"name": "adamw", "lr": 0.001, "weight_decay": 1.0,
     "extra": {"betas": [0.9, 0.98]}, "has_param_filter": false}
  ]
}
```

### Grid search folder layout

```
runs/grid/adamw/
├── lr0.001__weight_decay0.3/
│   ├── seed0/  {meta,history}.json + {model,optim}.pt
│   ├── seed1/
│   └── seed2/
├── lr0.001__weight_decay1/
│   └── ...
└── lr0.001__weight_decay3/
    └── ...
```

- One folder per HP combination, named by `_combo_label` (filesystem-safe).
- One subfolder per seed.
- Same artifacts as single-seed save.

→ You can navigate with `ls`, copy/move via `rsync`, inspect `meta.json`
without Python.

### Disk budget (rough estimate)

| Sweep size | Disk |
|---|---|
| 1 seed | ~4 MB |
| 5 seeds | ~20 MB |
| 9 combos × 5 seeds | ~180 MB |
| 30 combos × 5 seeds | ~600 MB |

`*.pt` files are in `.gitignore`. Don't commit them.

---

## The `history` dictionary

Logged every `eval_every` steps:

```python
history = {
    # Metadata
    'label':       str,
    'seed':        int,
    'specs_repr':  list[str],

    # Eval metrics (one entry per eval)
    'epoch':       list[int],
    'train_loss':  list[float],
    'test_loss':   list[float],
    'train_acc':   list[float],
    'test_acc':    list[float],
    'l2_norm':     list[float],   # sum_i ||W_i||^2

    # L2 decomposed by module group
    'l2_embed':    list[float],   # embed.W_E + pos_embed.W_pos
    'l2_attn':     list[float],   # W_K + W_Q + W_V + W_O
    'l2_mlp':      list[float],   # W_in + b_in + W_out + b_out
    'l2_unembed':  list[float],   # unembed.W_U

    # Optimizer dynamics (captured on the step before each eval)
    'grad_norm':           list[float|None],   # ||grad||
    'update_norm':         list[float|None],   # ||theta_after - theta_before||
    'update_norm_per_opt': list[list[float]|None],
    'eval_wallclock':      list[float],        # seconds since previous eval

    # Fourier progress measures (only if fourier_every is set)

    !!TOUT CA C'AI PAS ENCORE FAIT !!! MAIS ON A LES FCT DE NANDA (ET CLAUDE <3 <3 <3) DONC PAS TROP DUR A IMPLEMENTER!!
    'fourier_epoch':      list[int],
    'key_freqs':          list[list[int]],
    'restricted_loss':    list[float],
    'excluded_loss':      list[float],
    'gini_W_E':           list[float],
    'gini_W_L':           list[float],
}
```

Invariant: `len(history['epoch']) == len(history['train_loss']) == ...`

At epoch 0: `grad_norm`, `update_norm`, `update_norm_per_opt` are `None`
(no step has happened yet). `eval_wallclock[0]` is the init time, not a
training step time.

---

## Adding a new optimizer

The optimizer registry pattern means adding an algorithm is 3 lines.

### Example: adding Lion

```python
# 1. Import the class somewhere at the top of pipeline.py
from lion_pytorch import Lion

# 2. Add a builder function inside _register_default_optimizers()
def _build_lion(params, lr, weight_decay, **extra):
    return Lion(params, lr=lr, weight_decay=weight_decay, **extra)

# 3. Register it
_OPTIMIZER_REGISTRY['lion'] = _build_lion
```

That's it. You can now write:

```python
specs = [OptimizerSpec('lion', lr=3e-4, weight_decay=1.0,
                       extra={'beta1': 0.9, 'beta2': 0.99})]
```

→ `grid_search`, `run_multi_seed`, `save_run`, `from_run` all work without
any further modification.

### Builder contract

Every builder must have signature:

```python
def builder(params, lr, weight_decay, **extra) -> torch.optim.Optimizer:
    return SomeOptimizer(params, lr=lr, weight_decay=weight_decay, **extra)
```

The pipeline always calls it as
`registry[name](params, lr=spec.lr, weight_decay=spec.weight_decay, **spec.extra)`.

---

## Troubleshooting

### `OMP: Error #15: Initializing libomp.dylib` (macOS)

Add this to your shell or before launching Python:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

Inside Jupyter you usually don't see this.

### MPS doesn't support float64

The pipeline auto-detects MPS and falls back to float32 in
`cross_entropy_high_precision`. No action needed. On CUDA, float64 is
used for better stability at very low loss.

### `Parameter 'X' is not claimed by any OptimizerSpec`

Your `param_filter`s don't cover all parameters. Either widen one
filter, or add a catch-all spec:

```python
OptimizerSpec('adamw', lr=1e-3, weight_decay=1.0,
              extra={'betas': (0.9, 0.98)})   # param_filter=None
```

### Grid search reloads but won't continue training

`grid_search` is meant to start fresh runs OR reload finished ones.
It does not auto-extend a run with more epochs. To extend, reload manually:

```python
trainer = Trainer.from_run('runs/grid/adamw/lr0.001__weight_decay1/seed0',
                            specs=spec_builder(lr=1e-3, weight_decay=1.0))
trainer.fit(num_epochs=40_000)
trainer.save_run('runs/grid/adamw/lr0.001__weight_decay1/seed0')
```

### Importing pipeline.py raises `ModuleNotFoundError: optimizer`

You're running from the wrong directory. `cd` into `baseline/`:
```bash
cd baseline/
python -c "import pipeline; print('ok')"
```

In notebooks, make sure the notebook is opened from `baseline/`.

### `Trainer.from_run` raises `RuntimeError: Run at 'X' used param_filters`

The run was saved with hybrid specs (Muon / SOAP). Pass the specs explicitly:
```python
trainer = Trainer.from_run('runs/.../seedN', specs=spec_builder(...))
```

---

## Reference notebooks

| Notebook | What it does |
|---|---|
| `multiseed_baseline.ipynb` | AdamW × 5 seeds, the baseline reference |
| `grid_search.ipynb` | AdamW HP sweep (Nanda methodology) |
| `grid_search_muon_soap.ipynb` | Muon hybrid + SOAP hybrid HP sweeps |

Each notebook is self-contained and uses the same 2-phase methodology.
Start from `multiseed_baseline.ipynb` if you just want to reproduce
Nanda. Move to `grid_search.ipynb` to tune. Move to
`grid_search_muon_soap.ipynb` to compare optimizers.
