"""
viz_analysis.py
================

Helper module for multi-seed grokking analysis.

Functions :
    - load_seeds                : load history.json from each seed's dir
    - stack / stack_fourier     : stack a metric across seeds
    - epoch_mean_crosses        : mean-curve threshold crossing (visually aligned with bold mean)
    - compute_grokking_markers  : returns mem, circuit, grok dict (mean-curve crossings)
    - per_seed_stats            : per-seed grok stats (median, IQR, range)

Plot functions (all matplotlib, multi-seed band style) :
    - plot_curves_band          : 3 rows × 2 cols (loss / acc / L2, lin + log)
    - plot_fourier_losses_band  : 2 panels (train/test + excluded, + restricted)
    - plot_freq_mass_heatmap    : heatmap (epoch × freq) for W_L et W_E
    - plot_sparsity             : 6 panels (top5_conc, entropy, gini for WL and WE)
    - plot_fourier_components   : Nanda-style bar chart (W_E + W_U)
    - identify_key_freqs        : extract consensus key freqs from bar chart data
"""

from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch as t
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# =============================================================================
# Constants
# =============================================================================
COL_TRAIN, COL_TEST, COL_L2     = '#1f77b4', '#d62728', '#ff7f0e'
COL_EXCLUDED, COL_RESTRICTED    = '#2ca02c', '#9467bd'
COL_MEM, COL_CIRCUIT, COL_GROK  = '#1f77b4', '#9467bd', '#cc1f1a'


# =============================================================================
# Data loading
# =============================================================================
def _first_above(epochs, vals, thresh):
    for e, v in zip(epochs, vals):
        if v >= thresh: return e
    return None


def load_seeds(save_root, seeds, *, acc_thresh: float = 0.99):
    """Load seedN/history.json for each seed.

    Returns
    -------
    runs : list of {'seed', 'history', 'mem', 'grok'}
    runs_dict : {seed: history_dict}  — for compatibility with plot_seeds_band etc.
    """
    save_root = Path(save_root)
    runs, runs_dict = [], {}
    for s in seeds:
        p = save_root / f'seed{s}' / 'history.json'
        if not p.exists():
            print(f'  ⚠ skip seed {s} : {p} not found'); continue
        h = json.load(open(p))
        runs.append({
            'seed': s,
            'history': h,
            'mem':  _first_above(h['epoch'], h['train_acc'], acc_thresh),
            'grok': _first_above(h['epoch'], h['test_acc'],  acc_thresh),
        })
        runs_dict[s] = h
    return runs, runs_dict


# =============================================================================
# Stacking + aggregation
# =============================================================================
def _stack_curve(runs, key, time_key='epoch'):
    """Stack a metric across seeds. Handles different-length histories (caused by
    adaptive_logging: seeds that grok at different epochs have different snapshot counts)
    by interpolating to the UNION of all seeds' epochs. NaN where a seed has no data.

    Empty arrays (e.g. fourier_epoch when training was run with fourier_every=None) are skipped.
    """
    epoch_arrs, val_arrs = [], []
    for r in runs:
        h = r['history']
        if time_key not in h or key not in h: continue
        ep  = np.array(h[time_key])
        val = np.array(h[key])
        if ep.size == 0 or val.size == 0: continue   # skip seeds without this metric
        epoch_arrs.append(ep)
        val_arrs.append(val)
    if not val_arrs:
        return None, None
    # Union of all epochs sorted ascending
    all_epochs = np.array(sorted(set().union(*[set(ep.tolist()) for ep in epoch_arrs])))
    if all_epochs.size == 0:
        return None, None
    # Linear interpolation per seed, NaN outside the seed's actual range
    interp_vals = []
    for ep, val in zip(epoch_arrs, val_arrs):
        f = np.interp(all_epochs, ep, val, left=np.nan, right=np.nan)
        interp_vals.append(f)
    return all_epochs, np.stack(interp_vals, axis=0)


def stack(runs, key):           return _stack_curve(runs, key, time_key='epoch')
def stack_fourier(runs, key):   return _stack_curve(runs, key, time_key='fourier_epoch')


def _stack_freq_matrix(runs, key='wl_frequency_masses'):
    """Stack per-seed (n_snapshots × n_freqs) Fourier matrices on a common epoch grid
    via interpolation. Returns (epochs, mean_matrix) where mean_matrix has shape
    (n_epochs_union, n_freqs). Per-seed values are NaN outside their actual epoch range,
    then averaged via nanmean over seeds.

    Handles adaptive_logging correctly : seeds that grok at different epochs have
    different snapshot counts; their data is interpolated so the X-axis reflects
    the FULL training range (not truncated to the shortest seed).
    """
    raw_eps, raw_mats = [], []
    for r in runs:
        h = r['history']
        if not h.get('fourier_epoch') or key not in h or not h[key]: continue
        ep = np.array(h['fourier_epoch'])
        m  = np.array(h[key])
        if ep.size == 0 or m.size == 0: continue
        raw_eps.append(ep); raw_mats.append(m)
    if not raw_mats:
        return None, None
    # Union of all epochs (ascending)
    all_epochs = np.array(sorted(set().union(*[set(ep.tolist()) for ep in raw_eps])))
    if all_epochs.size == 0:
        return None, None
    n_freqs = raw_mats[0].shape[1]
    # Interpolate per seed × per freq onto the common grid
    interp = np.full((len(raw_mats), len(all_epochs), n_freqs), np.nan)
    for i, (ep, m) in enumerate(zip(raw_eps, raw_mats)):
        for j in range(n_freqs):
            interp[i, :, j] = np.interp(all_epochs, ep, m[:, j],
                                          left=np.nan, right=np.nan)
    with np.errstate(all='ignore'):
        mean_mat = np.nanmean(interp, axis=0)
    return all_epochs, mean_mat


def epoch_mean_crosses(runs, key, thresh, *, time_key='epoch'):
    """First epoch where the MEAN across seeds crosses `thresh` (aligned with bold mean curve).
    Uses nanmean to ignore NaN-padded entries from seeds with shorter history."""
    eps, stk = _stack_curve(runs, key, time_key=time_key)
    if eps is None: return None
    with np.errstate(all='ignore'):
        mean_curve = np.nanmean(stk, axis=0)
    for e, v in zip(eps, mean_curve):
        if not np.isnan(v) and v >= thresh: return float(e)
    return None


def compute_grokking_markers(runs, *, mem_thresh=0.99, grok_thresh=0.99,
                              circuit_key=None, circuit_thresh=0.5):
    """Returns dict with mem, circuit, grok markers.

    Each marker is the first epoch where the **MEAN curve across seeds** crosses the
    threshold. This definition is aligned with the bold mean curve plotted by
    `_plot_band` — same statistical quantity, no ambiguity.

        mem     : first epoch where mean(train_acc) >= mem_thresh
        circuit : first epoch where mean(<circuit_key>) >= circuit_thresh
        grok    : first epoch where mean(test_acc) >= grok_thresh

    If `circuit_key` is None (default), auto-detects :
      - 'wl_keyfreq_concentration' if present in any history (Phase 2 with fixed_key_freqs)
        — robust for diffuse spectra where top-5 alone is insufficient
      - else 'wl_top5_concentration' (Phase 1 / Nanda's canonical sparse case)
    """
    if circuit_key is None:
        has_keyfreq = any(r.get('history', {}).get('wl_keyfreq_concentration') for r in runs)
        circuit_key = 'wl_keyfreq_concentration' if has_keyfreq else 'wl_top5_concentration'
    return {
        'mem':     epoch_mean_crosses(runs, 'train_acc',  mem_thresh),
        'circuit': epoch_mean_crosses(runs, circuit_key,  circuit_thresh, time_key='fourier_epoch'),
        'grok':    epoch_mean_crosses(runs, 'test_acc',   grok_thresh),
    }


def per_seed_stats(runs):
    """Per-seed grok stats : median, IQR, range, raw list."""
    groks = [r['grok'] for r in runs if r['grok'] is not None]
    if not groks:
        return {'median': None, 'iqr': (None, None), 'range': (None, None), 'list': []}
    return {
        'median': float(np.median(groks)),
        'iqr':    (float(np.percentile(groks, 25)), float(np.percentile(groks, 75))),
        'range':  (min(groks), max(groks)),
        'list':   groks,
    }


# =============================================================================
# Plot helpers
# =============================================================================
def _plot_band(ax, epochs, stacks, color, label='',
               alpha_seed=0.25, alpha_band=0.15, lw_seed=0.7, lw_mean=2.6):
    """Multi-seed plot : transparent per-seed + min-max band + bold mean.

    Uses nan-aware aggregation so that seeds with shorter history (e.g. crashed early
    or interpolated from a shorter range) don't poison the band/mean.
    """
    if stacks is None or stacks.size == 0: return
    with np.errstate(all='ignore'):
        lo, hi, mu = np.nanmin(stacks, 0), np.nanmax(stacks, 0), np.nanmean(stacks, 0)
    ax.fill_between(epochs, lo, hi, color=color, alpha=alpha_band, zorder=2)
    for i in range(stacks.shape[0]):
        ax.plot(epochs, stacks[i], color=color, lw=lw_seed, alpha=alpha_seed, zorder=3)
    ax.plot(epochs, mu, color=color, lw=lw_mean, label=label, zorder=4)


def _add_vlines(ax, mem=None, circuit=None, grok=None, show_labels=True):
    if mem is not None:
        ax.axvline(mem, color=COL_MEM, ls=':', lw=1.6, alpha=0.75,
                   label=(f'mem = {int(mem)}' if show_labels else None))
    if circuit is not None:
        ax.axvline(circuit, color=COL_CIRCUIT, ls='--', lw=1.6, alpha=0.75,
                   label=(f'circuit = {int(circuit)}' if show_labels else None))
    if grok is not None:
        ax.axvline(grok, color=COL_GROK, ls='-.', lw=1.6, alpha=0.75,
                   label=(f'grok = {int(grok)}' if show_labels else None))


# =============================================================================
# Main plot functions
# =============================================================================
def plot_curves_band(runs, *, title='Training curves', markers=None,
                      save_path=None, figsize=(15, 13)):
    """3 rows × 2 cols : loss / accuracy / L2 norm, each in linear and log x-scale."""
    markers = markers or {}
    mem, circuit, grok = markers.get('mem'), markers.get('circuit'), markers.get('grok')

    ep_l, train_loss = stack(runs, 'train_loss')
    _,    test_loss  = stack(runs, 'test_loss')
    ep_a, train_acc  = stack(runs, 'train_acc')
    _,    test_acc   = stack(runs, 'test_acc')
    ep_w, l2_norm    = stack(runs, 'l2_norm')

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    for ax, scale in zip(axes[0], ['linear', 'log']):
        _plot_band(ax, ep_l, train_loss, COL_TRAIN, label='train loss')
        _plot_band(ax, ep_l, test_loss,  COL_TEST,  label='test loss')
        _add_vlines(ax, mem=mem, circuit=circuit, grok=grok)
        ax.set_yscale('log')
        if scale == 'log': ax.set_xscale('log')
        ax.set_xlabel('epoch'); ax.set_ylabel('cross entropy')
        ax.set_title(f'Loss — {scale} scale')
        ax.legend(loc='best', fontsize=8); ax.grid(True, which='both', alpha=0.3, linestyle=':')

    for ax, scale in zip(axes[1], ['linear', 'log']):
        _plot_band(ax, ep_a, train_acc, COL_TRAIN, label='train acc')
        _plot_band(ax, ep_a, test_acc,  COL_TEST,  label='test acc')
        _add_vlines(ax, mem=mem, circuit=circuit, grok=grok)
        ax.axhline(0.99, color='#444', ls=':', lw=0.8, alpha=0.5)
        if scale == 'log': ax.set_xscale('log')
        ax.set_xlabel('epoch'); ax.set_ylabel('accuracy')
        ax.set_title(f'Accuracy — {scale} scale')
        ax.set_ylim(-0.02, 1.04)
        ax.legend(loc='best', fontsize=8); ax.grid(True, alpha=0.3, linestyle=':')

    for ax, scale in zip(axes[2], ['linear', 'log']):
        if l2_norm is not None:
            _plot_band(ax, ep_w, l2_norm, COL_L2, label='||W||²')
        _add_vlines(ax, mem=mem, circuit=circuit, grok=grok)
        if scale == 'log': ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('epoch'); ax.set_ylabel('||W||²')
        ax.set_title(f'L2 norm — {scale} scale')
        ax.legend(loc='best', fontsize=8); ax.grid(True, which='both', alpha=0.3, linestyle=':')

    plt.suptitle(f'{title}   ·   n={len(runs)} seeds   ·   bold = mean, transparent = per-seed, band = min-max',
                 y=1.005, fontsize=13, fontweight='600')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_fourier_losses_band(runs, *, title='Fourier progress losses', markers=None,
                              excluded_key='excluded_all_loss_train',
                              restricted_key='restricted_loss_all',
                              save_path=None, figsize=(15, 5.5)):
    """2 panels (excluded | restricted Fourier loss) with train/test reference. Log-log axes.

    Nanda 2023 convention (from progress-measures-paper Grokking_Analysis.ipynb) :
      - excluded_loss : key freqs subtracted, evaluated on TRAIN only       -> excluded_all_loss_train
      - restricted_loss : projected onto key freqs, evaluated on ALL p*p   -> restricted_loss_all
    Pattern : excluded_train rises while restricted_all falls — they cross around grok.
    """
    markers = markers or {}
    mem, circuit, grok = markers.get('mem'), markers.get('circuit'), markers.get('grok')

    ep_f, excl      = stack_fourier(runs, excluded_key)
    _,    restr     = stack_fourier(runs, restricted_key)
    ep_l, train_loss = stack(runs, 'train_loss')
    _,    test_loss  = stack(runs, 'test_loss')

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[0]
    _plot_band(ax, ep_l, train_loss, COL_TRAIN, label='train loss')
    _plot_band(ax, ep_l, test_loss,  COL_TEST,  label='test loss')
    if excl is not None:
        _plot_band(ax, ep_f, excl, COL_EXCLUDED, label='excluded train (key freqs removed)')
    _add_vlines(ax, mem=mem, circuit=circuit, grok=grok)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('epoch'); ax.set_ylabel('cross entropy')
    ax.set_title('Excluded loss (Nanda — train set)')
    ax.legend(loc='best', fontsize=8); ax.grid(True, which='both', alpha=0.3, linestyle=':')

    ax = axes[1]
    _plot_band(ax, ep_l, train_loss, COL_TRAIN, label='train loss')
    _plot_band(ax, ep_l, test_loss,  COL_TEST,  label='test loss')
    if restr is not None:
        _plot_band(ax, ep_f, restr, COL_RESTRICTED, label='restricted all (key freqs only)')
    _add_vlines(ax, mem=mem, circuit=circuit, grok=grok)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('epoch'); ax.set_ylabel('cross entropy')
    ax.set_title('Restricted loss (Nanda — all data)')
    ax.legend(loc='best', fontsize=8); ax.grid(True, which='both', alpha=0.3, linestyle=':')

    plt.suptitle(f'{title}   ·   n={len(runs)} seeds   ·   bold = mean, band = min-max',
                 y=1.02, fontsize=12, fontweight='600')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_freq_mass_heatmap_per_seed(runs, *, title='Frequency mass evolution (per seed)',
                                     markers=None, save_path=None, figsize_per_seed=(13, 2.6)):
    """Per-seed heatmap (epoch × freq) grid : one row per seed, columns = W_L | W_E.

    No averaging across seeds — each row shows the true trajectory of one specific run.
    Useful when seeds have heterogeneous spectra (e.g. diffuse-solution configs like
    muon_m0_fast / egd_m0_fast where the mean would smear key freqs across seeds).
    """
    markers = markers or {}
    mem, circuit, grok = markers.get('mem'), markers.get('circuit'), markers.get('grok')

    valid = [r for r in runs if r.get('history', {}).get('fourier_epoch')
             and r['history'].get('wl_frequency_masses')]
    if not valid:
        print('plot_freq_mass_heatmap_per_seed: no Fourier snapshots'); return None

    n_seeds = len(valid)
    fig, axes = plt.subplots(n_seeds, 2,
                              figsize=(figsize_per_seed[0], figsize_per_seed[1] * n_seeds),
                              squeeze=False)

    for row_idx, r in enumerate(valid):
        h    = r['history']
        seed = r.get('seed', row_idx)
        ep   = np.array(h['fourier_epoch'])
        for col, (key, panel_label) in enumerate([
            ('wl_frequency_masses', 'W_L (neuron-logit)'),
            ('we_frequency_masses', 'W_E (embedding)'),
        ]):
            ax  = axes[row_idx, col]
            mat = np.array(h.get(key, []))
            if mat.size == 0:
                ax.text(0.5, 0.5, f'{key} N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='#999')
                ax.set_title(f'seed {seed} — {panel_label}', fontsize=10); continue
            vmin = max(1e-6, mat[mat > 0].min()) if (mat > 0).any() else 1e-6
            im = ax.imshow(mat.T, aspect='auto', cmap='magma', origin='lower',
                           extent=[ep[0], ep[-1], 1, mat.shape[1]],
                           norm=LogNorm(vmin=vmin, vmax=mat.max()))
            _add_vlines(ax, mem=mem, circuit=circuit, grok=grok, show_labels=(row_idx == 0))
            ax.set_xlabel('epoch' if row_idx == n_seeds - 1 else '')
            ax.set_ylabel('freq k')
            ax.set_title(f'seed {seed} — {panel_label}', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    plt.suptitle(f'{title}   ·   n={n_seeds} seeds (one row each)',
                 y=1.001, fontsize=12, fontweight='600')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_sparsity(runs, *, title='Spectral sparsity', markers=None,
                   save_path=None, figsize=(18, 9)):
    """6 panels : concentration (key_freqs if available, else top-5), entropy, gini for WL/WE.

    For the concentration metric : if `wl_keyfreq_concentration` exists in history (Phase 2 with
    fixed_key_freqs), uses that — measures mass in the FIXED key freqs. Else falls back to
    `wl_top5_concentration` which measures mass in top-5 of the current spectrum.
    """
    markers = markers or {}
    mem, circuit, grok = markers.get('mem'), markers.get('circuit'), markers.get('grok')

    # Choose concentration metric : prefer keyfreq if available
    has_keyfreq_wl = any(h.get('history', {}).get('wl_keyfreq_concentration') for h in runs)
    has_keyfreq_we = any(h.get('history', {}).get('we_keyfreq_concentration') for h in runs)
    wl_conc_key = 'wl_keyfreq_concentration' if has_keyfreq_wl else 'wl_top5_concentration'
    we_conc_key = 'we_keyfreq_concentration' if has_keyfreq_we else 'we_top5_concentration'
    wl_conc_title = 'WL — key_freqs mass concentration' if has_keyfreq_wl else 'WL — top-5 mass concentration'
    we_conc_title = 'WE — key_freqs mass concentration' if has_keyfreq_we else 'WE — top-5 mass concentration'

    metrics = [
        (wl_conc_key,             wl_conc_title,                   '#1f77b4'),
        (we_conc_key,             we_conc_title,                   '#2ca02c'),
        ('wl_entropy',            'WL — spectral entropy',         '#d62728'),
        ('we_entropy',            'WE — spectral entropy',         '#9467bd'),
        ('gini_W_L',              'WL — Gini coefficient',         '#17becf'),
        ('gini_W_E',              'WE — Gini coefficient',         '#ff7f0e'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()
    for ax, (key, panel_title, color) in zip(axes, metrics):
        ep_k, vals_k = stack_fourier(runs, key)
        if ep_k is None:
            ax.text(0.5, 0.5, f'{key} N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='#999'); continue
        _plot_band(ax, ep_k, vals_k, color, label=key)
        _add_vlines(ax, mem=mem, circuit=circuit, grok=grok)
        ax.set_xscale('log')
        ax.set_xlabel('epoch'); ax.set_ylabel(key.split('_')[-1])
        ax.set_title(panel_title, fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, which='both', alpha=0.3, linestyle=':')
    plt.suptitle(f'{title}   ·   n={len(runs)} seeds',
                 y=1.01, fontsize=13, fontweight='600')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.show()


# =============================================================================
# Nanda-style Fourier component bar chart
# =============================================================================
def plot_fourier_components(save_root, seeds, config, *,
                             title='Fourier components',
                             save_path=None, figsize=(16, 5),
                             color_cos='#ff7f0e', color_sin='#1f77b4',
                             show_seeds=True):
    """Bar chart Nanda-style : Fourier components of W_E and W_U (neuron-logit map).

    Loads `seedN/model.pt` for each seed, projects W_E and W_U on the real Fourier basis
    over Z_p, computes L2 norm of cos and sin components per frequency.

    Returns
    -------
    dict : {seed: {'we_cos','we_sin','wu_cos','wu_sin'}} of np.ndarray (length p//2)
    """
    from fourier_metrics import make_fourier_basis

    save_root = Path(save_root)
    p = config.p

    basis    = make_fourier_basis(config).cpu()
    cos_rows = basis[1::2]    # (p//2, p) : cos(1), cos(2), ...
    sin_rows = basis[2::2]    # (p//2, p) : sin(1), sin(2), ...
    n_freqs  = cos_rows.shape[0]
    freqs    = np.arange(1, n_freqs + 1)

    def _find_param(state, suffix):
        for k, v in state.items():
            if k.endswith(suffix): return v
        return None

    per_seed = {}
    for s in seeds:
        mp = save_root / f'seed{s}' / 'model.pt'
        if not mp.exists():
            print(f'  skip seed {s} : {mp} not found'); continue
        state = t.load(mp, map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        W_E   = _find_param(state, 'W_E')
        W_U   = _find_param(state, 'W_U')
        W_out = _find_param(state, 'W_out')
        if W_E is None or W_U is None or W_out is None:
            raise KeyError(f'W_E / W_U / W_out not found in {mp}. Keys: {list(state)}')
        # W_E projection along vocab axis
        W_E_p = W_E[:, :p].float().cpu()                         # (d_model, p)
        # W_L = neuron-logit map = W_U[:, :p].T @ W_out (cf. fourier_metrics.wl_frequency_masses)
        W_L   = (W_U[:, :p].float().cpu().T) @ W_out.float().cpu()  # (p, d_mlp)
        per_seed[s] = {
            'we_cos': (W_E_p @ cos_rows.T).norm(dim=0).numpy(),    # (p//2,) norm over d_model
            'we_sin': (W_E_p @ sin_rows.T).norm(dim=0).numpy(),
            'wl_cos': (cos_rows @ W_L).norm(dim=1).numpy(),         # (p//2,) norm over d_mlp
            'wl_sin': (sin_rows @ W_L).norm(dim=1).numpy(),
            # stashed raw matrices : needed by identify_key_freqs(method='permutation')
            '_W_L':       W_L.numpy(),                              # (p, d_mlp)
            '_basis_cos': cos_rows.numpy(),                         # (p//2, p)
            '_basis_sin': sin_rows.numpy(),                         # (p//2, p)
        }
    if not per_seed:
        print(f'plot_fourier_components: no seeds loaded from {save_root}'); return None

    def _agg(key):
        stk = np.array([per_seed[s][key] for s in per_seed])
        return stk.mean(0), stk.min(0), stk.max(0), stk

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panels = [
        ('we_cos', 'we_sin', 'Fourier Components of Embedding Matrix ($W_E$)'),
        ('wl_cos', 'wl_sin', 'Fourier Components of Neuron-Logit Map ($W_L = W_U^T W_{out}$)'),
    ]
    width = 0.42
    for ax, (key_c, key_s, panel_title) in zip(axes, panels):
        c_mean, c_min, c_max, c_stk = _agg(key_c)
        s_mean, s_min, s_max, s_stk = _agg(key_s)
        ax.bar(freqs - width/2, c_mean, width, yerr=[c_mean - c_min, c_max - c_mean],
               color=color_cos, alpha=0.85, edgecolor='black', linewidth=0.4,
               label='cos', capsize=0, error_kw={'lw': 0.7, 'alpha': 0.6})
        ax.bar(freqs + width/2, s_mean, width, yerr=[s_mean - s_min, s_max - s_mean],
               color=color_sin, alpha=0.85, edgecolor='black', linewidth=0.4,
               label='sin', capsize=0, error_kw={'lw': 0.7, 'alpha': 0.6})
        if show_seeds:
            for i in range(c_stk.shape[0]):
                ax.scatter(freqs - width/2, c_stk[i], color=color_cos, alpha=0.45, s=8, zorder=4)
                ax.scatter(freqs + width/2, s_stk[i], color=color_sin, alpha=0.45, s=8, zorder=4)
        ax.set_xlabel('Frequency k'); ax.set_ylabel('Norm of Fourier Component')
        ax.set_title(panel_title, fontsize=11)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.92)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    plt.suptitle(f'{title}   ·   n={len(per_seed)} seeds   ·   bar = mean, errbar = min-max, dots = per-seed',
                 y=1.02, fontsize=12, fontweight='600')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.show()
    return per_seed


def identify_key_freqs(
    per_seed_components,
    *,
    method: str = 'topk',            # 'topk' | 'cum_energy' | 'permutation'  (default = legacy, Phase 2 unchanged)
    tau: float = 0.90,               # for 'cum_energy'
    n_perms: int = 500,              # for 'permutation'
    alpha: float = 0.05,             # for 'permutation' : significance level
    correction: str = 'fdr',         # 'fdr' (Benjamini-Hochberg) | 'bonferroni' | None
    max_k: int = 20,                 # hard cap (all methods) — safety net
    k: int = 5,                      # for 'topk' (legacy)
    rng_seed: int = 42,              # for permutation reproducibility
    verbose: bool = True,
):
    """Extract per-seed key frequencies via one of three selection methods.

    Score per freq = wl_cos² + wl_sin² — uses **only the Neuron-Logit map W_L** (not W_E),
    consistent with `fourier_metrics.choose_key_freqs_from_wl` (Nanda's canonical method).

    Methods (from least to most rigorous)
    --------------------------------------
    'topk' (legacy)  : top-`k` freqs by W_L mass (Nanda's hardcoded k=5).
                       Pros : simple. Cons : fixed k ignores per-seed sparsity.

    'cum_energy'     : smallest k such that ∑ top-k masses ≥ `tau` × total mass.
                       Parseval-grounded — keeps freqs explaining `tau` fraction of W_L's
                       spectral energy. Default `tau=0.90`. Adapts to per-seed sparsity.

    'permutation'    : statistical test against a row-shuffle null of W_L (most rigorous).
                       For each seed, n_perms row-permutations of W_L define the null mass
                       distribution per freq. A freq is "key" if observed mass exceeds the
                       null at level `alpha`, with multiple-testing correction (`fdr` or
                       `bonferroni`). Requires `_W_L`, `_basis_cos`, `_basis_sin` stashed
                       by `plot_fourier_components` (already patched).

    All methods apply a hard cap `max_k` (selects top `max_k` by mass if exceeded).

    Returns
    -------
    dict with :
        'per_seed'  : {seed : sorted key freqs (1-indexed)}
        'consensus' : freqs selected in ALL seeds
        'majority'  : freqs selected in (n//2 + 1) seeds
        'method'    : the method used (for traceability)
        'info'      : per-seed extra info (n_keep, energy_captured, min_pval, …)
    """
    if per_seed_components is None or not per_seed_components:
        return {'per_seed': {}, 'consensus': [], 'majority': [],
                'method': method, 'info': {}}

    per_seed_kf = {}
    info = {}

    for s, comp in per_seed_components.items():
        wl_mass = comp['wl_cos']**2 + comp['wl_sin']**2     # (p//2,)
        n_freqs = len(wl_mass)

        if method == 'topk':
            order = np.argsort(wl_mass)[::-1]
            keep_idx = order[:min(k, max_k, n_freqs)]
            info[s] = {'n_keep': int(len(keep_idx)), 'k_param': k}

        elif method == 'cum_energy':
            order = np.argsort(wl_mass)[::-1]
            sorted_mass = wl_mass[order]
            total = float(sorted_mass.sum())
            if total <= 0:
                keep_idx = order[:1]
            else:
                cumfrac = np.cumsum(sorted_mass) / total
                k_keep = int(np.searchsorted(cumfrac, tau) + 1)
                k_keep = min(k_keep, max_k, n_freqs)
                keep_idx = order[:k_keep]
            captured = float(wl_mass[keep_idx].sum() / max(total, 1e-12))
            info[s] = {'n_keep': int(len(keep_idx)), 'tau': tau,
                       'energy_captured': round(captured, 4)}

        elif method == 'permutation':
            W_L       = comp.get('_W_L')
            cos_basis = comp.get('_basis_cos')
            sin_basis = comp.get('_basis_sin')
            if W_L is None or cos_basis is None or sin_basis is None:
                raise ValueError(
                    "method='permutation' requires per_seed_components[s] to contain "
                    "'_W_L', '_basis_cos', '_basis_sin'. Re-run plot_fourier_components "
                    "with the patched version, or use method='cum_energy'."
                )
            rng    = np.random.default_rng(int(rng_seed) + int(s))
            p_size = W_L.shape[0]
            # observed mass
            obs_cos = cos_basis @ W_L
            obs_sin = sin_basis @ W_L
            obs_mass = (obs_cos**2).sum(1) + (obs_sin**2).sum(1)
            # null distribution via row-shuffle of W_L
            null_mass = np.empty((n_perms, n_freqs))
            for i in range(n_perms):
                W_perm = W_L[rng.permutation(p_size)]
                c = cos_basis @ W_perm
                d = sin_basis @ W_perm
                null_mass[i] = (c**2).sum(1) + (d**2).sum(1)
            # per-freq p-value (one-sided, upper tail) with +1 pseudocount
            pvals = (np.sum(null_mass >= obs_mass[None, :], axis=0) + 1) / (n_perms + 1)
            # multiple-testing correction
            if correction == 'bonferroni':
                sig = pvals < (alpha / n_freqs)
            elif correction == 'fdr':
                order_p = np.argsort(pvals)
                ranked  = np.arange(1, n_freqs + 1)
                bh_thr  = alpha * ranked / n_freqs
                pvals_sorted = pvals[order_p]
                ok = pvals_sorted <= bh_thr
                sig = np.zeros(n_freqs, dtype=bool)
                if ok.any():
                    cutoff = int(np.max(np.where(ok)[0])) + 1
                    sig[order_p[:cutoff]] = True
            else:
                sig = pvals < alpha
            keep_idx = np.where(sig)[0]
            # cap : if too many significant, keep top-`max_k` by observed mass
            if len(keep_idx) > max_k:
                top_within = np.argsort(obs_mass[keep_idx])[::-1][:max_k]
                keep_idx = keep_idx[top_within]
            info[s] = {'n_keep': int(len(keep_idx)), 'alpha': alpha,
                       'correction': correction, 'n_perms': n_perms,
                       'min_pval': round(float(pvals.min()), 5)}

        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose : 'topk', 'cum_energy', 'permutation'."
            )

        per_seed_kf[s] = sorted((np.asarray(keep_idx) + 1).tolist())

    if verbose:
        print(f"[identify_key_freqs] method='{method}', max_k={max_k}")
        for s in sorted(per_seed_kf):
            kf = per_seed_kf[s]
            extra = info.get(s, {})
            extra_str = '  '.join(f"{k}={v}" for k, v in extra.items() if k != 'n_keep')
            print(f"  seed {s} : {len(kf):2d} key freqs = {kf}    ({extra_str})")

    counter = Counter()
    for kf in per_seed_kf.values():
        for f in kf:
            counter[f] += 1
    n_seeds   = len(per_seed_kf)
    consensus = sorted([f for f, c in counter.items() if c == n_seeds])
    majority  = sorted([f for f, c in counter.items() if c >= n_seeds // 2 + 1])

    return {
        'per_seed':  per_seed_kf,
        'consensus': consensus,
        'majority':  majority,
        'method':    method,
        'info':      info,
    }


def plot_fourier_components_per_seed(save_root, seeds, config, *,
                                      title='Per-seed Fourier components',
                                      save_path=None, row_height=2.3,
                                      color_cos='#ff7f0e', color_sin='#1f77b4'):
    """One row per seed showing W_E (left) and W_U (right) Fourier components.

    Useful to inspect seed-to-seed variability — do they find the same key freqs ?
    Returns the per-seed dict (same as plot_fourier_components).
    """
    from fourier_metrics import make_fourier_basis

    save_root = Path(save_root)
    p = config.p
    basis    = make_fourier_basis(config).cpu()
    cos_rows = basis[1::2]
    sin_rows = basis[2::2]
    n_freqs  = cos_rows.shape[0]
    freqs    = np.arange(1, n_freqs + 1)

    def _find_param(state, suffix):
        for k, v in state.items():
            if k.endswith(suffix): return v
        return None

    seeds_data = []
    for s in seeds:
        mp = save_root / f'seed{s}' / 'model.pt'
        if not mp.exists():
            print(f'  skip seed {s} : {mp} not found'); continue
        state = t.load(mp, map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        W_E   = _find_param(state, 'W_E')
        W_U   = _find_param(state, 'W_U')
        W_out = _find_param(state, 'W_out')
        if W_E is None or W_U is None or W_out is None:
            raise KeyError(f'W_E / W_U / W_out not found in {mp}.')
        W_E_p = W_E[:, :p].float().cpu()
        # W_L = neuron-logit map = W_U[:, :p].T @ W_out
        W_L   = (W_U[:, :p].float().cpu().T) @ W_out.float().cpu()
        seeds_data.append({
            'seed': s,
            'we_cos': (W_E_p @ cos_rows.T).norm(dim=0).numpy(),
            'we_sin': (W_E_p @ sin_rows.T).norm(dim=0).numpy(),
            'wl_cos': (cos_rows @ W_L).norm(dim=1).numpy(),
            'wl_sin': (sin_rows @ W_L).norm(dim=1).numpy(),
        })

    if not seeds_data:
        print(f'plot_fourier_components_per_seed: no seeds loaded'); return None

    n_seeds = len(seeds_data)
    fig, axes = plt.subplots(n_seeds, 2, figsize=(16, row_height * n_seeds))
    if n_seeds == 1: axes = axes.reshape(1, -1)
    width = 0.42

    for row, sd in enumerate(seeds_data):
        for col, (key_c, key_s, panel_label) in enumerate([
            ('we_cos', 'we_sin', 'Embedding $W_E$'),
            ('wl_cos', 'wl_sin', 'Neuron-Logit $W_L$'),
        ]):
            ax = axes[row, col]
            ax.bar(freqs - width/2, sd[key_c], width, color=color_cos, alpha=0.85,
                   edgecolor='black', linewidth=0.3, label='cos')
            ax.bar(freqs + width/2, sd[key_s], width, color=color_sin, alpha=0.85,
                   edgecolor='black', linewidth=0.3, label='sin')
            ax.set_title(f'seed {sd["seed"]} — {panel_label}', fontsize=10)
            ax.set_xlabel('Frequency k')
            ax.set_ylabel('Norm')
            ax.grid(True, alpha=0.3, axis='y', linestyle=':')
            if row == 0:
                ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f'{title}   ·   n={n_seeds} seeds', y=1.005, fontsize=13, fontweight='600')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.show()

    return {sd['seed']: {k: v for k, v in sd.items() if k != 'seed'} for sd in seeds_data}


# =============================================================================
# 3D frequency mass surface plot
# =============================================================================
def plot_freq_mass_heatmap_3d_per_seed(runs, *, key='wl_frequency_masses',
                                         title='Frequency mass — 3D surface (per seed)',
                                         markers=None, save_path=None,
                                         log_z=True, log_x=False,
                                         figsize=(14, 8), elev=30, azim=-60,
                                         cmap='magma'):
    """3D surface plot per seed (one figure per seed) : (epoch × freq) → mass.

    No averaging — each figure shows the true trajectory of one specific run.
    Vertical lines at mem/circuit/grok markers shown as dashed segments on the floor.
    `save_path` : if given, suffixed with `_seed{s}` per figure.
    """
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers '3d')
    markers = markers or {}

    valid = [r for r in runs if r.get('history', {}).get('fourier_epoch')
             and r['history'].get(key)]
    if not valid:
        print(f'plot_freq_mass_heatmap_3d_per_seed: no data for {key}'); return

    for r in valid:
        h    = r['history']
        seed = r.get('seed', '?')
        ep   = np.array(h['fourier_epoch'])
        mat  = np.array(h[key])
        if mat.size == 0:
            print(f'  seed {seed} : empty {key} — skip'); continue

        freqs = np.arange(1, mat.shape[1] + 1)
        X, Y  = np.meshgrid(ep, freqs)
        Z = mat.T
        if log_z:
            Z = np.log10(np.maximum(np.nan_to_num(Z, nan=1e-10), 1e-10))

        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none',
                                alpha=0.92, linewidth=0, antialiased=True,
                                rstride=1, cstride=1)
        ax.set_xlabel('epoch'); ax.set_ylabel('frequency index k')
        ax.set_zlabel('log10(mass)' if log_z else 'mass')
        if log_x:
            ax.set_xscale('log')
        ax.view_init(elev=elev, azim=azim)

        z_floor = Z.min()
        for label_m, ep_v, color in [
            ('mem',     markers.get('mem'),     '#1f77b4'),
            ('circuit', markers.get('circuit'), '#9467bd'),
            ('grok',    markers.get('grok'),    '#cc1f1a'),
        ]:
            if ep_v is None: continue
            ax.plot([ep_v, ep_v], [1, mat.shape[1]], [z_floor, z_floor],
                    color=color, lw=2.2, ls='--', label=f'{label_m}={int(ep_v)}')
        if any(markers.get(k) is not None for k in ('mem', 'circuit', 'grok')):
            ax.legend(loc='upper left', fontsize=9, framealpha=0.85)

        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08,
                      label='log10(mass)' if log_z else 'mass')
        ax.set_title(f'{title} — seed {seed}', fontsize=12, pad=12, fontweight='600')
        plt.tight_layout()
        if save_path is not None:
            sp = str(save_path)
            if '.' in sp.rsplit('/', 1)[-1]:
                stem, ext = sp.rsplit('.', 1)
                sp = f'{stem}_seed{seed}.{ext}'
            else:
                sp = f'{sp}_seed{seed}.png'
            plt.savefig(sp, bbox_inches='tight', facecolor='white')
        plt.show()


# =============================================================================
# 3D frequency mass — Plotly interactive version
# =============================================================================
def plot_freq_mass_heatmap_3d_plotly_per_seed(runs, *, key='wl_frequency_masses',
                                                title='Frequency mass — 3D surface (interactive, per seed)',
                                                markers=None, save_path=None,
                                                log_z=True, log_x=False,
                                                width=1100, height=750,
                                                colorscale='Magma'):
    """3D interactive Plotly surface plot, one figure per seed.

    No averaging — each figure shows the true trajectory of one specific run.
    Rotate / zoom / pan with the mouse. Hover for exact (epoch, k, mass) values.
    `save_path` : if given, suffixed with `_seed{s}` per figure (.html for interactive).
    """
    import plotly.graph_objects as go
    markers = markers or {}

    valid = [r for r in runs if r.get('history', {}).get('fourier_epoch')
             and r['history'].get(key)]
    if not valid:
        print(f'plot_freq_mass_heatmap_3d_plotly_per_seed: no data for {key}'); return []

    figures = []
    for r in valid:
        h    = r['history']
        seed = r.get('seed', '?')
        ep   = np.array(h['fourier_epoch'])
        mat  = np.array(h[key])
        if mat.size == 0:
            print(f'  seed {seed} : empty {key} — skip'); continue

        freqs = np.arange(1, mat.shape[1] + 1)
        Z = mat.T
        if log_z:
            Z_display = np.log10(np.maximum(np.nan_to_num(Z, nan=1e-10), 1e-10))
            z_label = 'log10(mass)'
        else:
            Z_display = np.nan_to_num(Z, nan=0.0)
            z_label = 'mass'

        fig = go.Figure()
        fig.add_trace(go.Surface(
            z=Z_display, x=ep, y=freqs,
            colorscale=colorscale,
            colorbar=dict(title=z_label, len=0.8),
            hovertemplate=('epoch=%{x:.0f}<br>freq k=%{y}<br>'
                            + z_label + '=%{z:.3f}<extra></extra>'),
        ))

        z_floor = float(Z_display.min())
        z_ceil  = float(Z_display.max())
        for label_m, ep_v, color in [
            ('mem',     markers.get('mem'),     'blue'),
            ('circuit', markers.get('circuit'), 'purple'),
            ('grok',    markers.get('grok'),    'red'),
        ]:
            if ep_v is None: continue
            fig.add_trace(go.Scatter3d(
                x=[ep_v, ep_v, ep_v], y=[1, mat.shape[1]/2, mat.shape[1]],
                z=[z_floor, (z_floor+z_ceil)/2, z_ceil],
                mode='lines', name=f'{label_m}={int(ep_v)}',
                line=dict(color=color, width=5, dash='dash'),
                hovertemplate=f'{label_m}=%{{x:.0f}}<extra></extra>',
            ))

        fig.update_layout(
            title=dict(text=f'{title} — seed {seed}', font=dict(size=14)),
            scene=dict(
                xaxis=dict(title='epoch', type='log' if log_x else 'linear'),
                yaxis=dict(title='frequency index k'),
                zaxis=dict(title=z_label),
                camera=dict(eye=dict(x=1.6, y=1.5, z=1.0)),
                aspectratio=dict(x=1.5, y=1.0, z=0.8),
            ),
            width=width, height=height,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.85)'),
        )

        if save_path is not None:
            sp = str(save_path)
            if '.' in sp.rsplit('/', 1)[-1]:
                stem, ext = sp.rsplit('.', 1)
                sp = f'{stem}_seed{seed}.{ext}'
            else:
                sp = f'{sp}_seed{seed}.html'
            if sp.endswith('.html'):
                fig.write_html(sp)
            else:
                try: fig.write_image(sp)
                except Exception as e:
                    print(f'  write_image failed ({e}) — saving HTML instead')
                    fig.write_html(sp.rsplit('.', 1)[0] + '.html')

        fig.show()
        figures.append(fig)
    return figures
