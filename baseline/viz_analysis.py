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
    arrays, epochs_ref = [], None
    for r in runs:
        h = r['history']
        if time_key not in h or key not in h: continue
        ep, val = np.array(h[time_key]), np.array(h[key])
        if epochs_ref is None: epochs_ref = ep
        else:
            n = min(len(epochs_ref), len(ep))
            epochs_ref = epochs_ref[:n]
            arrays = [a[:n] for a in arrays]; val = val[:n]
        arrays.append(val)
    return (epochs_ref, np.stack(arrays, axis=0)) if arrays else (None, None)


def stack(runs, key):           return _stack_curve(runs, key, time_key='epoch')
def stack_fourier(runs, key):   return _stack_curve(runs, key, time_key='fourier_epoch')


def epoch_mean_crosses(runs, key, thresh, *, time_key='epoch'):
    """First epoch where the MEAN across seeds crosses `thresh` (aligned with bold mean curve)."""
    eps, stk = _stack_curve(runs, key, time_key=time_key)
    if eps is None: return None
    mean_curve = stk.mean(axis=0)
    for e, v in zip(eps, mean_curve):
        if v >= thresh: return float(e)
    return None


def compute_grokking_markers(runs, *, mem_thresh=0.99, grok_thresh=0.99,
                              circuit_key='wl_top5_concentration', circuit_thresh=0.5):
    """Returns dict with mem, circuit, grok markers.

    Each marker is the first epoch where the **MEAN curve across seeds** crosses the
    threshold. This definition is aligned with the bold mean curve plotted by
    `_plot_band` — same statistical quantity, no ambiguity.

        mem     : first epoch where mean(train_acc) >= mem_thresh
        circuit : first epoch where mean(wl_top5_concentration) >= circuit_thresh
        grok    : first epoch where mean(test_acc) >= grok_thresh
    """
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
    """Multi-seed plot : transparent per-seed + min-max band + bold mean."""
    if stacks is None or stacks.size == 0: return
    ax.fill_between(epochs, stacks.min(0), stacks.max(0), color=color, alpha=alpha_band, zorder=2)
    for i in range(stacks.shape[0]):
        ax.plot(epochs, stacks[i], color=color, lw=lw_seed, alpha=alpha_seed, zorder=3)
    ax.plot(epochs, stacks.mean(0), color=color, lw=lw_mean, label=label, zorder=4)


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
                              restricted_key='restricted_loss_test',
                              save_path=None, figsize=(15, 5.5)):
    """2 panels (excluded | restricted Fourier loss) with train/test reference. Log-log axes."""
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
        _plot_band(ax, ep_f, excl, COL_EXCLUDED, label='excluded (key freqs removed)')
    _add_vlines(ax, mem=mem, circuit=circuit, grok=grok)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('epoch'); ax.set_ylabel('cross entropy')
    ax.set_title('Train/test + excluded Fourier loss')
    ax.legend(loc='best', fontsize=8); ax.grid(True, which='both', alpha=0.3, linestyle=':')

    ax = axes[1]
    _plot_band(ax, ep_l, train_loss, COL_TRAIN, label='train loss')
    _plot_band(ax, ep_l, test_loss,  COL_TEST,  label='test loss')
    if restr is not None:
        _plot_band(ax, ep_f, restr, COL_RESTRICTED, label='restricted (key freqs only)')
    _add_vlines(ax, mem=mem, circuit=circuit, grok=grok)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('epoch'); ax.set_ylabel('cross entropy')
    ax.set_title('Train/test + restricted Fourier loss')
    ax.legend(loc='best', fontsize=8); ax.grid(True, which='both', alpha=0.3, linestyle=':')

    plt.suptitle(f'{title}   ·   n={len(runs)} seeds   ·   bold = mean, band = min-max',
                 y=1.02, fontsize=12, fontweight='600')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_freq_mass_heatmap(runs, *, title='Frequency mass evolution', markers=None,
                            save_path=None, figsize=(16, 6)):
    """Heatmap (epoch × freq) for W_L and W_E frequency masses. Magma colormap, log scale."""
    markers = markers or {}
    mem, circuit, grok = markers.get('mem'), markers.get('circuit'), markers.get('grok')

    def _freq_mass_matrix(runs, key):
        matrices, epochs_ref = [], None
        for r in runs:
            h = r['history']
            if not h.get('fourier_epoch') or key not in h or not h[key]: continue
            ep = np.array(h['fourier_epoch'])
            m = np.array(h[key])
            if epochs_ref is None: epochs_ref = ep
            else:
                n = min(len(epochs_ref), len(ep))
                epochs_ref = epochs_ref[:n]
                matrices = [a[:n] for a in matrices]; m = m[:n]
            matrices.append(m)
        return (epochs_ref, np.stack(matrices, axis=0).mean(0)) if matrices else (None, None)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (key, panel_title) in zip(axes, [
        ('wl_frequency_masses', 'W_L (unembedding) — mean over seeds'),
        ('we_frequency_masses', 'W_E (embedding) — mean over seeds'),
    ]):
        ep_k, mat = _freq_mass_matrix(runs, key=key)
        if mat is None:
            ax.text(0.5, 0.5, f'{key} N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='#999')
            ax.set_title(panel_title); continue
        im = ax.imshow(mat.T, aspect='auto', cmap='magma', origin='lower',
                       extent=[ep_k[0], ep_k[-1], 1, mat.shape[1]],
                       norm=LogNorm(vmin=max(1e-6, mat[mat>0].min()), vmax=mat.max()))
        _add_vlines(ax, mem=mem, circuit=circuit, grok=grok)
        ax.set_xlabel('epoch'); ax.set_ylabel('frequency index k')
        ax.set_title(panel_title)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        plt.colorbar(im, ax=ax, label='frequency mass (log)')
    plt.suptitle(f'{title}   ·   horizontal bands = key freqs',
                 y=1.02, fontsize=13, fontweight='600')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_sparsity(runs, *, title='Spectral sparsity', markers=None,
                   save_path=None, figsize=(18, 9)):
    """6 panels : wl/we top5_concentration, wl/we entropy, gini_W_L/W_E. Log-x."""
    markers = markers or {}
    mem, circuit, grok = markers.get('mem'), markers.get('circuit'), markers.get('grok')
    metrics = [
        ('wl_top5_concentration', 'WL — top-5 mass concentration', '#1f77b4'),
        ('we_top5_concentration', 'WE — top-5 mass concentration', '#2ca02c'),
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
        W_E = _find_param(state, 'W_E')
        W_U = _find_param(state, 'W_U')
        if W_E is None or W_U is None:
            raise KeyError(f'W_E or W_U not found in {mp}. Keys: {list(state)}')
        W_E_p = W_E[:, :p].float().cpu()
        W_U_p = W_U[:, :p].float().cpu()
        per_seed[s] = {
            'we_cos': (W_E_p @ cos_rows.T).norm(dim=0).numpy(),
            'we_sin': (W_E_p @ sin_rows.T).norm(dim=0).numpy(),
            'wu_cos': (W_U_p @ cos_rows.T).norm(dim=0).numpy(),
            'wu_sin': (W_U_p @ sin_rows.T).norm(dim=0).numpy(),
        }
    if not per_seed:
        print(f'plot_fourier_components: no seeds loaded from {save_root}'); return None

    def _agg(key):
        stk = np.array([per_seed[s][key] for s in per_seed])
        return stk.mean(0), stk.min(0), stk.max(0), stk

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panels = [
        ('we_cos', 'we_sin', 'Fourier Components of Embedding Matrix ($W_E$)'),
        ('wu_cos', 'wu_sin', 'Fourier Components of Neuron-Logit Map ($W_U$)'),
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


def identify_key_freqs(per_seed_components, *, k=5):
    """Extract the top-k key frequencies per seed from `plot_fourier_components` output.

    Score per freq = sum of |we_cos| + |we_sin| + |wu_cos| + |wu_sin| across both panels.
    Then top-k across each seed.

    Returns
    -------
    dict with :
        'per_seed'  : {seed : sorted top-k freqs}
        'consensus' : freqs in ALL seeds' top-k
        'majority'  : freqs in (n_seeds//2 + 1) seeds' top-k
    """
    if per_seed_components is None:
        return {'per_seed': {}, 'consensus': [], 'majority': []}

    per_seed_topk = {}
    for s, comp in per_seed_components.items():
        combined = comp['we_cos'] + comp['we_sin'] + comp['wu_cos'] + comp['wu_sin']
        top_k = np.argsort(combined)[-k:][::-1] + 1   # +1 : freqs are 1-indexed
        per_seed_topk[s] = sorted(top_k.tolist())

    counter = Counter()
    for freqs_list in per_seed_topk.values():
        for f in freqs_list:
            counter[f] += 1
    n_seeds   = len(per_seed_topk)
    consensus = sorted([f for f, c in counter.items() if c == n_seeds])
    majority  = sorted([f for f, c in counter.items() if c >= n_seeds // 2 + 1])

    return {
        'per_seed':  per_seed_topk,
        'consensus': consensus,
        'majority':  majority,
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
        W_E = _find_param(state, 'W_E')
        W_U = _find_param(state, 'W_U')
        if W_E is None or W_U is None:
            raise KeyError(f'W_E or W_U not found in {mp}.')
        W_E_p = W_E[:, :p].float().cpu()
        W_U_p = W_U[:, :p].float().cpu()
        seeds_data.append({
            'seed': s,
            'we_cos': (W_E_p @ cos_rows.T).norm(dim=0).numpy(),
            'we_sin': (W_E_p @ sin_rows.T).norm(dim=0).numpy(),
            'wu_cos': (W_U_p @ cos_rows.T).norm(dim=0).numpy(),
            'wu_sin': (W_U_p @ sin_rows.T).norm(dim=0).numpy(),
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
            ('wu_cos', 'wu_sin', 'Neuron-Logit $W_U$'),
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
