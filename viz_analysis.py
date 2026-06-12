

from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
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

    plt.suptitle(f'{title} n={len(runs)} seeds',
                 y=1.005, fontsize=13, fontweight='600')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_fourier_losses_band(runs, *, title='Fourier progress losses', markers=None,
                              excluded_key='excluded_all_loss_train',
                              restricted_key='restricted_loss_all',
                              save_path=None, figsize=None,
                              show_circuit=True, show_title=True,
                              shared_legend=False, share_axes=False,
                              show_accuracy=False,
                              label_fontsize=12, tick_fontsize=10,
                              title_fontsize=13, suptitle_fontsize=14,
                              legend_fontsize=10):

    markers = markers or {}
    mem, circuit, grok = markers.get('mem'), markers.get('circuit'), markers.get('grok')
    circuit_to_draw = circuit if show_circuit else None

    ep_f, excl      = stack_fourier(runs, excluded_key)
    _,    restr     = stack_fourier(runs, restricted_key)
    ep_l, train_loss = stack(runs, 'train_loss')
    _,    test_loss  = stack(runs, 'test_loss')
    if show_accuracy:
        ep_a, train_acc = stack(runs, 'train_acc')
        _,    test_acc  = stack(runs, 'test_acc')

    n_panels = 3 if show_accuracy else 2
    if figsize is None:
        figsize = (18.5, 5.2) if show_accuracy else (13, 5.2)

    rc = {
        'font.size':         label_fontsize,
        'axes.titlesize':    title_fontsize,
        'axes.labelsize':    label_fontsize,
        'xtick.labelsize':   tick_fontsize,
        'ytick.labelsize':   tick_fontsize,
        'legend.fontsize':   legend_fontsize,
        'axes.spines.top':   False,
        'axes.spines.right': False,
    }

    def _decorate(ax, panel_title, ylabel, is_left):
        _add_vlines(ax, mem=mem, circuit=circuit_to_draw, grok=grok)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('$log_{10}(epoch)$')
        if is_left or not share_axes or ylabel != 'cross-entropy loss':
            ax.set_ylabel(ylabel)
        ax.set_title(panel_title, fontsize=title_fontsize)
        if not shared_legend:
            ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, which='major', alpha=0.25, linestyle='-',  linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.12, linestyle=':',  linewidth=0.4)

    with plt.rc_context(rc):
        # When accuracy is shown, y-scales differ between loss and acc panels :
        # only share x in that case. Otherwise behave as before.
        sharey_val = share_axes and not show_accuracy
        fig, axes = plt.subplots(1, n_panels, figsize=figsize,
                                  sharex=share_axes, sharey=sharey_val)
        # Panel layout: when accuracy is shown, [Accuracy | Excluded | Restricted];
        # otherwise [Excluded | Restricted].
        panel_order = [1, 2, 0] if show_accuracy else [0, 1]
        ax = axes[panel_order[0]]
        _plot_band(ax, ep_l, train_loss, COL_TRAIN, label='train loss')
        _plot_band(ax, ep_l, test_loss,  COL_TEST,  label='test loss')
        if excl is not None:
            _plot_band(ax, ep_f, excl, COL_EXCLUDED, label='excluded loss')
        _decorate(ax, 'Excluded loss', 'cross-entropy loss', is_left=True)

        ax = axes[panel_order[1]]
        _plot_band(ax, ep_l, train_loss, COL_TRAIN, label='train loss')
        _plot_band(ax, ep_l, test_loss,  COL_TEST,  label='test loss')
        if restr is not None:
            _plot_band(ax, ep_f, restr, COL_RESTRICTED, label='restricted loss')
        _decorate(ax, 'Restricted loss', 'cross-entropy loss', is_left=False)

        if show_accuracy:
            ax = axes[panel_order[2]]
            _plot_band(ax, ep_a, train_acc, COL_TRAIN, label='train acc')
            _plot_band(ax, ep_a, test_acc,  COL_TEST,  label='test acc')
            _decorate(ax, 'Accuracy', 'accuracy', is_left=False)

        if title and show_title:
            fig.suptitle(title, y=1.005, fontsize=suptitle_fontsize, fontweight='600')

        if shared_legend:
            seen, handles, labels = set(), [], []
            for ax in axes:
                for h, l in zip(*ax.get_legend_handles_labels()):
                    if l and l not in seen:
                        seen.add(l); handles.append(h); labels.append(l)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            fig.legend(handles, labels, loc='center left',
                       bbox_to_anchor=(1.0, 0.5), frameon=True, framealpha=0.9)
            plt.tight_layout(rect=[0, 0, 0.88 if show_accuracy else 0.86, 1])
        else:
            plt.tight_layout()

        if save_path is not None:
            sp = Path(save_path)
            plt.savefig(sp, bbox_inches='tight', facecolor='white', dpi=300)
            if sp.suffix.lower() == '.png':
                plt.savefig(sp.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        plt.show()


def plot_freq_mass_heatmap_per_seed(runs, *, title='Frequency mass evolution (per seed)',
                                     markers=None, save_path=None, figsize_per_seed=(13, 2.6),
                                     show_title=True,
                                     label_fontsize=12, tick_fontsize=10,
                                     title_fontsize=10, suptitle_fontsize=12,
                                     legend_fontsize=9):
    markers = markers or {}
    mem, circuit, grok = markers.get('mem'), markers.get('circuit'), markers.get('grok')

    valid = [r for r in runs if r.get('history', {}).get('fourier_epoch')
             and r['history'].get('wl_frequency_masses')]
    if not valid:
        print('plot_freq_mass_heatmap_per_seed: no Fourier snapshots'); return None

    n_seeds = len(valid)
    rc = {
        'font.size':       label_fontsize,
        'axes.labelsize':  label_fontsize,
        'axes.titlesize':  title_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'legend.fontsize': legend_fontsize,
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(n_seeds, 2,
                                  figsize=(figsize_per_seed[0], figsize_per_seed[1] * n_seeds),
                                  squeeze=False)

        for row_idx, r in enumerate(valid):
            h    = r['history']
            seed = r.get('seed', row_idx)
            ep   = np.array(h['fourier_epoch'])
            for col, (key, panel_label) in enumerate([
                ('we_frequency_masses', '$W_E$ (embedding)'),
                ('wl_frequency_masses', '$W_L$ (neuron-logit)'),
            ]):
                ax  = axes[row_idx, col]
                mat = np.array(h.get(key, []))
                if mat.size == 0:
                    ax.text(0.5, 0.5, f'{key} N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=tick_fontsize, color='#999')
                    ax.set_title(f'{panel_label}', fontsize=title_fontsize)
                    continue
                vmin = max(1e-6, mat[mat > 0].min()) if (mat > 0).any() else 1e-6
                im = ax.imshow(mat.T, aspect='auto', cmap='magma', origin='lower',
                               extent=[ep[0], ep[-1], 1, mat.shape[1]],
                               norm=LogNorm(vmin=vmin, vmax=mat.max()))
                _add_vlines(ax, mem=mem, circuit=circuit, grok=grok,
                            show_labels=(row_idx == 0))
                ax.set_xlabel('epoch' if row_idx == n_seeds - 1 else '',
                              fontsize=label_fontsize)
                ax.set_ylabel('frequencies', fontsize=label_fontsize)
                ax.set_title(f'{panel_label}', fontsize=title_fontsize)
                cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
                cb.ax.tick_params(labelsize=tick_fontsize)
                if row_idx == 0:
                    ax.legend(loc='best', fontsize=legend_fontsize, framealpha=0.85)

        if show_title:
            plt.suptitle(f'{title}   ·   n={n_seeds} seeds (one row each)',
                         y=1.001, fontsize=suptitle_fontsize, fontweight='600')
        plt.tight_layout()
        if save_path is not None:
            sp = Path(save_path)
            plt.savefig(sp, bbox_inches='tight', facecolor='white', dpi=300)
            if sp.suffix.lower() == '.png':
                plt.savefig(sp.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        plt.show()


def plot_sparsity(runs, *, title='Spectral sparsity', markers=None,
                   save_path=None, figsize=(18, 9)):
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
def load_fourier_components(save_root, seeds, config):
    """Load model.pt per seed and project W_E, W_L on the Fourier basis."""
    from fourier_metrics import make_fourier_basis

    save_root = Path(save_root)
    p = config.p
    basis    = make_fourier_basis(config).cpu()
    cos_rows = basis[1::2]
    sin_rows = basis[2::2]

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
        W_E_p = W_E[:, :p].float().cpu()
        W_L   = (W_U[:, :p].float().cpu().T) @ W_out.float().cpu()
        per_seed[s] = {
            'we_cos':     (W_E_p @ cos_rows.T).norm(dim=0).numpy(),
            'we_sin':     (W_E_p @ sin_rows.T).norm(dim=0).numpy(),
            'wl_cos':     (cos_rows @ W_L).norm(dim=1).numpy(),
            'wl_sin':     (sin_rows @ W_L).norm(dim=1).numpy(),
            '_W_L':       W_L.numpy(),
            '_basis_cos': cos_rows.numpy(),
            '_basis_sin': sin_rows.numpy(),
        }
    return per_seed


def converged_seeds(save_root, seeds, thresh=0.99):
    """Return seeds whose final test_acc reaches thresh, read from history.json."""
    save_root = Path(save_root)
    out = []
    for s in seeds:
        hp = save_root / f'seed{s}' / 'history.json'
        if not hp.exists(): continue
        h = json.loads(hp.read_text())
        if h.get('test_acc') and h['test_acc'][-1] >= thresh:
            out.append(s)
    return out


def plot_fourier_components(save_root, seeds, config, *,
                             title='Fourier components',
                             save_path=None, figsize=(16, 5),
                             color_cos='#ff7f0e', color_sin='#1f77b4',
                             show_seeds=True):
    per_seed = load_fourier_components(save_root, seeds, config)
    if not per_seed:
        print(f'plot_fourier_components: no seeds loaded from {save_root}'); return None

    n_freqs = len(next(iter(per_seed.values()))['we_cos'])
    freqs   = np.arange(1, n_freqs + 1)

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
    method: str = 'topk',            
    tau: float = 0.90,               
    n_perms: int = 500,            
    alpha: float = 0.05,             
    correction: str = 'fdr',         
    max_k: int = 20,               
    k: int = 5,                      
    rng_seed: int = 42,              
    verbose: bool = True,
):
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
    per_seed = load_fourier_components(save_root, seeds, config)
    if not per_seed:
        print(f'plot_fourier_components_per_seed: no seeds loaded'); return None

    n_freqs    = len(next(iter(per_seed.values()))['we_cos'])
    freqs      = np.arange(1, n_freqs + 1)
    seeds_data = [{'seed': s, **per_seed[s]} for s in per_seed]
    n_seeds    = len(seeds_data)
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


# =============================================================================
# Grid-search heatmap
# =============================================================================
def heatmap_panel(ax, df, x_key, y_key, title, *,
                  value='eg_med', fixed=None, log=True,
                  vmin=None, vmax=None, cmap='viridis_r', fmt='.0f',
                  cbar_label=None):
    """One heatmap panel of a hyperparameter grid result.
    """
    import seaborn as sns

    sub = df if fixed is None else df.query(' and '.join(f'{k}=={v}' for k, v in fixed.items()))
    piv = sub.pivot(index=y_key, columns=x_key, values=value).sort_index(ascending=False)
    vals = np.log10(piv.values) if log else piv.values
    tested = set(map(tuple, sub[[y_key, x_key]].values))
    untested = np.array([[(y, x) not in tested for x in piv.columns] for y in piv.index])
    nogrok = (np.isnan(piv.values) & ~untested
              if value == 'eg_med'
              else np.zeros_like(piv.values, dtype=bool))
    mask = untested | nogrok
    if vmin is None: vmin = np.nanmin(vals)
    if vmax is None: vmax = np.nanmax(vals)
    label = cbar_label or (f'log10({value})' if log else value)
    sns.heatmap(vals, ax=ax, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax,
                annot=piv.values, fmt=fmt, cbar=True,
                cbar_kws={'label': label},
                xticklabels=[f'{c:g}' for c in piv.columns],
                yticklabels=[f'{i:g}' for i in piv.index],
                linewidths=0.5, linecolor='white', annot_kws={'fontsize': 9})
    for i, j in zip(*np.where(nogrok)):
        ax.text(j + 0.5, i + 0.5, '✗', ha='center', va='center',
                color='#aa3333', fontsize=14, fontweight='bold')
    for i, j in zip(*np.where(untested)):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                    facecolor='#f5f5f5', edgecolor='white', lw=0.5))
        ax.text(j + 0.5, i + 0.5, '·', ha='center', va='center',
                color='#999999', fontsize=14)
    ax.set_title(title); ax.set_xlabel(x_key); ax.set_ylabel(y_key)


def plot_grid_grokking_step(panels, *, value='eg_med', log=True,
                             suptitle=None, save_path=None,
                             figsize=None, panel_width=5.2, panel_height=4.5):
    """Render a single-row multi-panel `heatmap_panel` grid for a hp sweep.

    Parameters
    ----------
    panels   : list of dicts, each with keys
                 'df'    : aggregated grid dataframe
                 'x_key' : column for x axis
                 'y_key' : column for y axis
                 'title' : panel title
                 'fixed' : optional dict of column->value filters
                 (extra keys are forwarded to heatmap_panel)
    value    : column passed to each panel (default 'eg_med').
    log      : log10 color scale (default True).
    suptitle : optional figure-level title.
    save_path : if given, saves .png at dpi=200 and a sibling .pdf.
    """
    n = len(panels)
    if figsize is None:
        figsize = (panel_width * n, panel_height)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, panels):
        kw = {k: v for k, v in p.items()
              if k not in ('df', 'x_key', 'y_key', 'title')}
        kw.setdefault('value', value)
        kw.setdefault('log', log)
        heatmap_panel(ax, p['df'], p['x_key'], p['y_key'], p['title'], **kw)
    if suptitle:
        plt.suptitle(suptitle, y=1.02, fontsize=13, fontweight='600')
    plt.tight_layout()
    if save_path is not None:
        sp = Path(save_path)
        plt.savefig(sp, bbox_inches='tight', facecolor='white', dpi=200)
        if sp.suffix.lower() == '.png':
            plt.savefig(sp.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.show()


# =============================================================================
# Per-seed marker extraction + synthesis figures
# =============================================================================
def _interp_crossing(epochs, values, thresh, *, above=True):
    epochs = np.asarray(epochs, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = (v >= thresh) if above else (v <= thresh)
    idx = np.argmax(ok) if ok.any() else None
    if idx is None or idx == 0:
        return epochs[0] if (ok.any() and idx == 0) else np.nan
    e0, e1, v0, v1 = epochs[idx - 1], epochs[idx], v[idx - 1], v[idx]
    if v1 == v0:
        return e1
    frac = (thresh - v0) / (v1 - v0)
    return e0 + frac * (e1 - e0)


def _extract_seed_markers(save_root, seed):
    """Compute every per-seed timing / dynamics marker from a single history.json."""
    p = Path(save_root) / f'seed{seed}' / 'history.json'
    if not p.exists():
        return None
    h = json.load(open(p))
    ep  = np.asarray(h['epoch'], float)
    fep = np.asarray(h['fourier_epoch'], float)
    out = {}

    out['T_mem']        = _interp_crossing(ep,  h['train_acc'], 0.99)
    out['T_grok']       = _interp_crossing(ep,  h['test_acc'],  0.99)
    out['T_circuit']    = _interp_crossing(fep, h['wl_keyfreq_concentration'], 0.5)
    out['T_circuit_25'] = _interp_crossing(fep, h['wl_keyfreq_concentration'], 0.25)
    out['T_circuit_75'] = _interp_crossing(fep, h['wl_keyfreq_concentration'], 0.75)
    out['T_circuit_90'] = _interp_crossing(fep, h['wl_keyfreq_concentration'], 0.90)
    out['T_test_50']    = _interp_crossing(ep,  h['test_acc'],  0.5)

    exc = np.asarray(h['excluded_all_loss_train'], float)
    i_min = int(np.nanargmin(exc))
    out['exc_min']   = exc[i_min]
    out['T_exc_min'] = fep[i_min]
    rec = np.nan
    for i in range(i_min, len(exc)):
        if exc[i] >= 1.0:
            rec = fep[i]; break
    out['T_exc_recover'] = rec
    ent = np.nan
    for i in range(i_min, -1, -1):
        if exc[i] >= 1.0:
            ent = fep[i]; break
    out['T_exc_enter'] = ent

    res = np.asarray(h['restricted_loss_all'], float)
    out['T_res_01']  = _interp_crossing(fep, res, 0.1, above=False)
    out['res_final'] = float(np.nanmedian(res[-10:]))

    l2 = np.asarray(h['l2_norm'], float)
    i_pk = int(np.nanargmax(l2))
    out['l2_peak']   = l2[i_pk]
    out['T_l2_peak'] = ep[i_pk]
    out['l2_final']  = float(np.nanmedian(l2[-10:]))
    out['l2_init']   = l2[0]

    gini = np.asarray(h['gini_W_L'], float)
    out['gini_final'] = float(np.nanmedian(gini[-10:]))
    out['gini_max']   = float(np.nanmax(gini))
    out['T_gini_05']  = _interp_crossing(fep, gini, 0.5)

    out['n_keys'] = len(h['key_freqs'][-1])

    masses = np.asarray(h['wl_frequency_masses'], float)
    kf     = [int(k) for k in h['key_freqs'][-1]]
    emerg  = np.asarray([
        _interp_crossing(fep, masses[:, k - 1], 4.0 / 56.0) for k in kf
    ], float)
    out['emerg_first']  = np.nanmin(emerg) if len(emerg) else np.nan
    out['emerg_last']   = np.nanmax(emerg) if len(emerg) else np.nan
    out['emerg_spread'] = out['emerg_last'] - out['emerg_first']

    out['wl_kf_conc_final'] = float(np.nanmedian(
        np.asarray(h['wl_keyfreq_concentration'], float)[-10:]))
    return out


def compute_marker_csv(configs, seeds, csv_path):
    """Build the per-seed markers DataFrame and write it to csv_path.

    configs : list of dicts with keys 'label', 'save_root', 'family', 'momentum', 'regime'.
    """
    rows = []
    for cfg in configs:
        for s in seeds:
            r = _extract_seed_markers(cfg['save_root'], s)
            if r is None:
                print(f'missing: {cfg["label"]} seed{s}'); continue
            r.update(dict(config=cfg['label'], family=cfg['family'],
                          momentum=cfg['momentum'], regime=cfg['regime'], seed=s))
            rows.append(r)
    df = pd.DataFrame(rows)
    df['gap']           = df['T_grok'] - df['T_mem']
    df['gap_ratio']     = df['T_grok'] / df['T_mem']
    df['circuit_lead']  = df['T_grok'] - df['T_circuit']
    df['cleanup_dur']   = df['T_exc_recover'] - df['T_exc_min']
    df['dip_width']     = df['T_exc_recover'] - df['T_exc_enter']
    df['mem_to_excmin'] = df['T_exc_min'] - df['T_mem']

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def plot_pr_width(configs, markers_csv, save_path, *, seeds=range(5)):
    """Effective circuit width: PR(t) of W_L freq masses on a grok-normalized log grid."""
    mk = pd.read_csv(markers_csv).set_index(['config', 'seed'])
    fig, ax = plt.subplots(figsize=(10.5, 6))
    grid = np.logspace(-2.2, np.log10(60), 400)
    for cfg in configs:
        curves = []
        for s in seeds:
            hp = Path(cfg['save_root']) / f'seed{s}' / 'history.json'
            if not hp.exists(): continue
            h = json.load(open(hp))
            fep = np.asarray(h['fourier_epoch'], float)
            M   = np.asarray(h['wl_frequency_masses'], float)
            M   = M / M.sum(axis=1, keepdims=True)
            PR  = 1.0 / np.sum(M**2, axis=1)
            try:
                tg = mk.loc[(cfg['label'], s), 'T_grok']
            except KeyError:
                continue
            x = fep / tg
            keep = x > 0
            curves.append(np.interp(grid, x[keep], PR[keep], left=np.nan, right=np.nan))
        if not curves: continue
        mean = np.nanmean(np.vstack(curves), axis=0)
        ax.plot(grid, mean, color=cfg['color'], lw=1.9, label=cfg['label'],
                ls='-' if 'fast' in cfg['label'] else '--')
    ax.axvline(1.0, color='k', lw=1.2, alpha=0.7)
    ax.text(1.04, 50, '$T_{grok}$', fontsize=11)
    ax.set_xscale('log')
    ax.set_xlabel('epoch / $T_{grok}$ (per seed)')
    ax.set_ylabel('participation ratio of $W_L$ freq masses\n'
                  '(effective # of active frequencies)')
    ax.set_title('Effective circuit width along the (grok-normalized) trajectory'
                 ' — mean over 5 seeds')
    ax.grid(alpha=0.3, ls=':')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, facecolor='white')
    plt.show()


def plot_tcircuit_vs_tgrok(configs, markers_csv, save_path):
    """Per-seed scatter of T_circuit vs T_grok with the diagonal ordering guide."""
    df = pd.read_csv(markers_csv)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for cfg in configs:
        sub = df[df.config == cfg['label']]
        ax.scatter(sub['T_grok'], sub['T_circuit'], color=cfg['color'], s=55,
                   marker='o' if 'fast' in cfg['label'] else 's', label=cfg['label'],
                   edgecolor='k', linewidth=0.4, zorder=3)
    lims = [80, 30000]
    ax.plot(lims, lims, 'k-', lw=1, alpha=0.6)
    ax.fill_between(lims, lims, [lims[1]] * 2, color='red', alpha=0.05)
    ax.text(110, 16000, 'circuit concentrates\nAFTER grokking',
            fontsize=10, color='darkred')
    ax.text(2500, 200, 'classic order:\ncircuit before grokking',
            fontsize=10, color='darkgreen')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('$T_{grok}$ (test acc >= 0.99)')
    ax.set_ylabel('$T_{circuit}$ ($W_L$ key-freq concentration >= 0.5)')
    ax.set_title('Per-seed circuit-formation vs grokking time')
    ax.grid(alpha=0.3, ls=':')
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, facecolor='white')
    plt.show()
