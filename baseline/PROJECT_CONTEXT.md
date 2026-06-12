# Project Context — Grokking Analysis Pipeline

## Goal

Multi-seed training and Fourier-based mechanistic analysis of the grokking phenomenon
(Nanda et al. 2023, *"Progress measures for grokking via mechanistic interpretability"*)
on the modular addition task `(a + b) mod p`, comparing how different optimizers reach
the Fourier circuit.

---

## Model architecture

Defined in `model.py` (`Config` dataclass). Small decoder-only transformer following Nanda's setup:

- `p = 113` (prime modulus), vocab size = `p + 1` (extra `=` token)
- `d_model = 128`, `d_mlp = 512`, `num_heads = 4`, `n_ctx = 3`
- 1 transformer block, ReLU activation
- Input sequence: `(a, b, =)` ; output: prediction of `(a+b) mod p`
- `frac_train = 0.3` (30 % of all `p×p` pairs as train, rest as test)
- Adaptive logging flags: `adaptive_logging`, `adaptive_logging_thresh`, `adaptive_logging_factor`

---

## Optimizers

Hybrid setup: main optimizer for internal 2D weights, **AdamW** for embeddings + biases.

- `param_filter` for "internal 2D" : `lambda n, p: p.ndim >= 2 and 'embed' not in n`
- **EGD** (SVD mode): `extra={'momentum': m, 'mode': 'svd'}`
- **Muon** (Newton–Schulz): `extra={'momentum': m}`
- **AdamW**: `extra={'betas': (0.9, 0.98)}` — β₂ = 0.98 (Nanda's choice, not Adam's 0.999)
- In every hybrid, the AdamW pair is fixed to `lr=1e-3, wd=1.0, betas=(0.9, 0.98)`
- For pure AdamW: single `OptimizerSpec`, no filter

---

## Training pipeline (`pipeline.py`)

- **`Trainer`** (single-seed): takes `Config`, `list[OptimizerSpec]`, `seed`, `label`,
  `eval_every`, `fourier_every`, `fixed_key_freqs` (optional), `warmup_steps`.
- **`run_multi_seed`**: trains N seeds for one config, resume-aware (skips seeds whose
  `history.json` already exists).
- **`grid_search`**: Cartesian product over a `param_grid × seeds`, resume-aware.
- **`aggregate_grid`**: reduces grid_search results to a sorted table `{params + stats}`.
- **`adaptive_logging`**: once `mean(train_acc) >= adaptive_logging_thresh`,
  `eval_every` and `fourier_every` are both multiplied by `adaptive_logging_factor`
  (`= 5` in all notebooks).
- Per-seed output dir contains: `history.json` (curves), `model.pt`, `optim.pt`, `meta.json`.

---

## Fourier module (`fourier_metrics.py`, Nanda's conventions)

- Real Fourier basis over `Z_p`: row 0 = DC, rows `2k-1 / 2k` = `cos_k / sin_k` for
  `k = 1..p//2 = 56`.
- `W_E` projected on vocab axis → `embedding_frequency_masses`.
- `W_L = W_U[:, :p].T @ W_out` (neuron-logit map composition) → `wl_frequency_masses`.
- `choose_key_freqs_from_wl(top_k, rel_threshold)`: selects key freqs from final `W_L`.
- `fourier_metrics(..., key_freqs=...)`: per-snapshot computation. If `key_freqs` is None,
  uses `choose_key_freqs_from_wl`; if given, uses the provided fixed list.
- Stored in `history.json` per snapshot:
  - `restricted_loss_{train, test, all}`, `restricted_acc_*`
  - `excluded_all_loss_{train, test, all}`, `excluded_all_acc_*`
  - `excluded_loss` (per-freq list, train) and `excluded_loss_mean`
  - `wl_top5_concentration`, `we_top5_concentration`
  - `wl_keyfreq_concentration`, `we_keyfreq_concentration` (only when `key_freqs` is set)
  - `wl_frequency_masses`, `we_frequency_masses` (full length-56 vectors)
  - `wl_entropy`, `we_entropy`, `gini_W_L`, `gini_W_E`
  - `key_freqs` (the list actually used at that snapshot)

---

## Analysis protocol — Phase 1 / Phase 2

Per-optimizer notebooks: `egd_phase2.ipynb`, `muon_phase2.ipynb`, `adamw_phase2.ipynb`.

**Common settings**:

```python
SEEDS            = [0, 1, 2, 3, 4]
NUM_EPOCHS       = 25_000
EVAL_EVERY       = 10
FOURIER_EVERY    = 10
WARMUP_STEPS     = 10
CONVERGED_THRESH = 0.99
BASE_CONFIG = Config(..., adaptive_logging=True,
                          adaptive_logging_thresh=0.99,
                          adaptive_logging_factor=5)
```

### Section 1.1 — Phase 1 training (NO Fourier metrics)

`run_multi_seed` with the optimizer hybrid for each `CONFIG ∈ CONFIGS`, resume-aware,
`fourier_every=None`. Saved to `runs/phase2_optim/{config_name}/seed{s}/`.

### Section 1.2 — Reload + `plot_curves_band` (per CONFIG)

Markers: `compute_grokking_markers(runs)` with `circuit` forced to `None` in Phase 1
(no Fourier history available).

### Section 1.3 — Identify key freqs from final `model.pt`

`plot_fourier_components(save_root, seeds, config)` returns per-seed components
(`we_cos`, `we_sin`, `wl_cos`, `wl_sin`, plus stashed `_W_L`, `_basis_cos`, `_basis_sin`).
Then `identify_key_freqs(per_seed, k=5)` →
`cfg['per_seed_freqs']` (per-seed list of 5 freqs each), `cfg['key_freqs_majority']`.

### Section 1.4 — Diagnostic comparison (read-only, does NOT affect Phase 2)

For each config, recomputes per-seed components inline, then applies 5 methods:
`topk(k=5)`, `cum_energy(τ=0.85, 0.90, 0.95)`, `permutation(n_perms=500, α=0.05, FDR)`.
Results stored in `cfg['kf_diagnostics']`. Produces two pandas tables.

### Section 2.1 — Phase 2 training with fixed per-seed key freqs

Loops over seeds, creates
`Trainer(..., fixed_key_freqs=cfg['per_seed_freqs'][s], fourier_every=10)`.
Saved to `runs/phase2_optim/{config_name}_fixedkf/seed{s}/`.

### Section 2.2 — Phase 2 plots

- `plot_fourier_losses_band` — 2 panels, **strict Nanda convention** (from `progress-measures-paper/Grokking_Analysis.ipynb`):
  `excluded_key='excluded_all_loss_train'` (key freqs subtracted, eval on TRAIN),
  `restricted_key='restricted_loss_all'` (projected onto key freqs, eval on FULL p*p grid).
- `plot_freq_mass_heatmap_per_seed` — per-seed grid `n_seeds × (W_L | W_E)`,
  **no averaging across seeds**.
- `plot_sparsity` — 6 panels (concentration WL/WE, entropy WL/WE, gini WL/WE).
  Auto-detects `wl_keyfreq_concentration` when present, else falls back to top-5.
- (Optional) `plot_freq_mass_heatmap_3d_per_seed` and `*_plotly_per_seed`.

---

## Flexible Phase 2 — `phase2_v2_flexible.ipynb`

Standalone notebook for configs whose `k=5` selection is sub-optimal.

**Constants**:

```python
KF_METHOD   = 'cum_energy'
KF_TAU      = 0.90
KF_MAX_K    = 25
PHASE1_ROOT = runs/phase2_optim       # read-only
SAVE_ROOT   = runs/phase2_v2_flexkf   # new dir
```

**Cells**:

1. Identify per-seed key freqs from Phase 1 `model.pt` using
   `identify_key_freqs(method='cum_energy', tau=0.90, max_k=25)`.
2. Train Phase 2 with these flexible per-seed `key_freqs` (resume-aware).
3. Plots — same set as Section 2.2 ; the auto `circuit_key` detection picks up
   `wl_keyfreq_concentration`.
4. Numeric comparison v1 (`k=5`) vs v2 (flex) on `wl_keyfreq_concentration`,
   `restricted_loss_train`, `n_keys`.

---

## Standalone read-only analysis — `key_freqs_analysis.ipynb`

Loads Phase 1 final `model.pt` for all 10 configs (EGD ×4, Muon ×4, AdamW ×2),
applies 5 selection methods (`topk-5`, `cum_t0.85/0.90/0.95`, `permutation FDR`)
and produces 3 tables:

- per-seed details
- per-config × method summary (`n_keep` mean ± std)
- colored compact heatmap (red ≥ 2 vs k=5, yellow ≥ 1)

No training, no writes under `runs/`.

---

## Grid-search notebook — `grid_search_runner.ipynb`

`SEEDS = [0, 1]` (2 seeds — note that "median" stats with `n=2` coincide with the mean).

```python
PARAM_GRID_EGD   = {'lr_egd':   [1e-3, 1e-2, 0.1, 0.3, 1.0],
                    'wd_egd':   [0,    1e-3, 1e-2, 0.1, 1.0],
                    'momentum_egd': [0, 0.9]}
PARAM_GRID_MUON  = {'lr_muon':  [1e-3, 1e-2, 0.1, 0.3, 1.0],
                    'wd_muon':  [0,    1e-3, 1e-2, 0.1, 1.0]}  # momentum hardcoded
PARAM_GRID_ADAMW = {'lr':       [...],
                    'weight_decay': [...]}                    # betas hardcoded
```

Per-config history aggregated via `aggregate_grid` → DataFrame columns include:
`eg_med` (`epoch_grok_median`), `eg_min`, `eg_max`, `mem_med`, `gap_med`,
`l2_final_med`, `final_test_acc_med`.

Visualizations in `grid_views.ipynb`: `1×5` heatmap row of `log10(eg_med)` over `(lr, wd)`.

---

## Key-freq identification methods (`viz_analysis.identify_key_freqs`)

Score per freq = `wl_cos**2 + wl_sin**2` (uses only `W_L`, not `W_E` — matches
`choose_key_freqs_from_wl`).

| Method | Description | Key params |
|---|---|---|
| **`topk`** *(default)* | top-`k` by `W_L` mass. `k=5` reproduces Nanda exactly. | `k` |
| **`cum_energy`** | smallest `k` such that `Σ top-k masses ≥ τ × total mass` (Parseval-grounded). | `tau` (default `0.90`) |
| **`permutation`** | row-shuffle null of `W_L`, `n_perms` permutations, per-freq p-value, multiple-testing correction. | `n_perms`, `alpha`, `correction` (`'fdr'` BH or `'bonferroni'`) |

All methods apply `max_k` cap (default `20`). Verbose print per seed lists chosen freqs
and method-specific stats.

---

## Grokking markers (`viz_analysis.compute_grokking_markers`)

Mean-curve threshold crossings:

| Marker | Definition | Default threshold |
|---|---|---|
| **`mem`** | first epoch where `mean(train_acc)` crosses `mem_thresh` | `0.99` |
| **`circuit`** | first epoch where `mean(<circuit_key>)` crosses `circuit_thresh` | `0.5` |
| **`grok`** | first epoch where `mean(test_acc)` crosses `grok_thresh` | `0.99` |

`circuit_key` auto-detects: uses `'wl_keyfreq_concentration'` if any history contains it
(Phase 2 with `fixed_key_freqs`), else `'wl_top5_concentration'`.

---

## Viz and animation modules

### `viz_analysis.py`

- `load_seeds`, `compute_grokking_markers`, `per_seed_stats`
- `plot_curves_band`, `plot_fourier_losses_band`
- `plot_freq_mass_heatmap_per_seed` (2D, `n_seeds × 2` grid)
- `plot_freq_mass_heatmap_3d_per_seed` (matplotlib, one fig per seed, auto-suffix `_seed{s}`)
- `plot_freq_mass_heatmap_3d_plotly_per_seed` (interactive HTML, one per seed)
- `plot_sparsity` (6 panels)
- `plot_fourier_components`, `plot_fourier_components_per_seed`
- `identify_key_freqs` (3-method dispatch)

**Note**: no averaging-over-seeds heatmap is exposed anymore.

### `freq_animation.py`

- `animate_freq_mass_plotly(runs, seed=None, ...)` → interactive 3D growing-surface HTML
- `animate_freq_mass_mpl(runs, seed=None, ...)` → MP4/GIF (auto fallback to GIF if no ffmpeg)

Both single-seed only ; `seed=None` auto-picks the first valid seed with a printed note.

---

## Directory layout

> **Note**: capital `F` in `Fourier`.

```
/Users/mverest/Desktop/Fourier/Opti_ML/baseline/
  pipeline.py, model.py, fourier_metrics.py, viz_analysis.py, freq_animation.py
  optimizer/ (egd.py, muon.py, soap.py)
  runs/
    grid_clean/{egd,muon,muon_no_mom,adamw}/    ← grid-search results
    phase2_optim/{config}/                       ← Phase 1 training (no Fourier)
    phase2_optim/{config}_fixedkf/               ← Phase 2 v1 (topk-5 fixed)
    phase2_v2_flexkf/{config}/                   ← Phase 2 v2 (cum_t0.90 flex)
  figs_phase2_optim/, figs_phase2_v2/            ← saved plots
  Notebooks:
    egd_phase2.ipynb, muon_phase2.ipynb, adamw_phase2.ipynb   (Phase 1+2, k=5)
    phase2_v2_flexible.ipynb                                 (flex Phase 2 for selected configs)
    key_freqs_analysis.ipynb                                 (read-only comparison across all configs)
    grid_search_runner.ipynb, grid_views.ipynb               (HP sweep)
```

---

## Constraints

- **Do not modify anything inside `runs/`** (training artifacts).
- **Always use the capital-`F` path**: `/Users/mverest/Desktop/Fourier/Opti_ML/baseline`.
  The shell `cwd` `/Users/mverest/Desktop/Opti_ML` is the git checkout but not the active
  working copy.
