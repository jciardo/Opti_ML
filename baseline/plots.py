from __future__ import annotations
import numpy as np
import torch as t
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# =============================================================================
# Training curves
# =============================================================================

def plot_curves(history: dict, title: str = "Training run") -> None:
    """Plot loss and accuracy curves on a 2x2 grid (linear x + log x).
    """
    colors = {"train": "#636EFA", "test": "#EF553B"}
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Loss (linear x)", "Accuracy (linear x)",
                        "Loss (log x)",    "Accuracy (log x)"),
        vertical_spacing=0.15,
    )

    for row_idx, x_type in [(1, "linear"), (2, "log")]:
        show_legend = (row_idx == 1)
        # Loss column (col=1)
        for key, name in [("train_loss", "train"), ("test_loss", "test")]:
            fig.add_trace(
                go.Scatter(
                    x=history["epoch"], y=history[key], mode="lines",
                    name=name, line=dict(color=colors[name]),
                    legendgroup=name, showlegend=show_legend,
                ),
                row=row_idx, col=1,
            )
        # Accuracy column (col=2)
        for key, name in [("train_acc", "train"), ("test_acc", "test")]:
            fig.add_trace(
                go.Scatter(
                    x=history["epoch"], y=history[key], mode="lines",
                    name=name, line=dict(color=colors[name]),
                    legendgroup=name, showlegend=False,
                ),
                row=row_idx, col=2,
            )
        x_label = "epoch" if row_idx == 1 else "epoch (log)"
        fig.update_xaxes(type=x_type, title_text=x_label, row=row_idx, col=1)
        fig.update_xaxes(type=x_type, title_text=x_label, row=row_idx, col=2)
        fig.update_yaxes(type="log", title_text="cross entropy", row=row_idx, col=1)
        fig.update_yaxes(range=[0, 1.02], title_text="accuracy", row=row_idx, col=2)

    fig.update_layout(title=title, template="plotly_white", width=950, height=720)
    fig.show()


def plot_fourier_loss_comparison(
    item,
    title: str = "Nanda Fourier progress losses",
    *,
    excluded_key: str = "excluded_all_loss_train",
    restricted_key: str = "restricted_loss_test",
) -> None:
    """Plot train/test loss against excluded and restricted Fourier losses.

    Normal train/test losses are logged at `epoch`; Fourier losses are logged at
    `fourier_epoch`, so the traces intentionally use different x arrays.
    """
    history = _get_history(item)
    if not history.get("fourier_epoch"):
        print("plot_fourier_loss_comparison: no Fourier snapshots found")
        return

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Train/test + {excluded_key}",
            f"Train/test + {restricted_key}",
        ),
        horizontal_spacing=0.10,
    )

    base_traces = [
        ("train_loss", "train loss", "#636EFA"),
        ("test_loss", "test loss", "#EF553B"),
    ]
    for col in (1, 2):
        for key, name, color in base_traces:
            fig.add_trace(
                go.Scatter(
                    x=history["epoch"],
                    y=history[key],
                    mode="lines",
                    name=name,
                    line=dict(color=color),
                    legendgroup=name,
                    showlegend=(col == 1),
                ),
                row=1, col=col,
            )

    fig.add_trace(
        go.Scatter(
            x=history["fourier_epoch"],
            y=history[excluded_key],
            mode="lines+markers",
            name=excluded_key,
            line=dict(color="#00CC96", width=3),
            legendgroup=excluded_key,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=history["fourier_epoch"],
            y=history[restricted_key],
            mode="lines+markers",
            name=restricted_key,
            line=dict(color="#AB63FA", width=3),
            legendgroup=restricted_key,
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_xaxes(title_text="epoch", row=1, col=2)
    fig.update_yaxes(type="log", title_text="cross entropy", row=1, col=1)
    fig.update_yaxes(type="log", title_text="cross entropy", row=1, col=2)
    fig.update_layout(title=title, template="plotly_white", width=1100, height=430)
    fig.show()


# =============================================================================
#  Learned modular addition table
# =============================================================================

@t.no_grad()
def plot_final_grid(trainer, title: str = "Learned modular addition table") -> None:
    """Visualize the modular addition table learned by the model.
    """
    p = trainer.config.p
    logits = trainer.model(trainer.all_data)[:, -1, :p]
    preds  = logits.argmax(dim=-1).detach().cpu().reshape(p, p).numpy()
    labels = ((trainer.all_data[:, 0] + trainer.all_data[:, 1]) % p)
    labels = labels.detach().cpu().reshape(p, p).numpy()

    # Boolean mask of train pairs in (P, P) space
    train_mask = np.zeros((p, p), dtype=float)
    for x, y, _ in trainer.train_pairs:
        train_mask[x, y] = 1.0

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("prediction", "wrong cells", "train split"),
    )
    fig.add_trace(
        go.Heatmap(z=preds, colorscale="Viridis", showscale=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=(preds != labels).astype(float), colorscale="Reds", showscale=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Heatmap(z=train_mask, colorscale="Greys", showscale=False),
        row=1, col=3,
    )
    fig.update_xaxes(title_text="y")
    fig.update_yaxes(title_text="x", autorange="reversed")
    fig.update_layout(title=title, template="plotly_white", width=950, height=360)
    fig.show()


# =============================================================================
# Multi-seed plots (one Trainer per seed)
# =============================================================================

def _get_history(item):
    """Accept either a Trainer or a history dict; return the dict."""
    return item.history if hasattr(item, 'history') else item


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip('#')
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"


_AGGREGATE_ALIASES = {
    'median':   'median',
    'mean':     'mean',
    'min_max':  'min_max',
    'max_min':  'min_max',     # alias for convenience
    'minmax':   'min_max',
    'envelope': 'min_max',
    'band':     'min_max',
}


def _normalize_aggregate(name: str) -> str:
    """Map any accepted alias to one of {'median', 'mean', 'min_max'}."""
    if name not in _AGGREGATE_ALIASES:
        raise ValueError(
            f"aggregate must be one of {sorted(set(_AGGREGATE_ALIASES))}, "
            f"got {name!r}"
        )
    return _AGGREGATE_ALIASES[name]


def plot_seeds_overlay(
    results: dict,
    title: str = "Multi-seed overlay",
    alpha: float = 0.35,
    agg_width: int = 3,
    aggregate: str = 'median',
    include_l2_norm: bool = True,
) -> None:
    """Multi-seed plot — combines loss, accuracy and ||W||² in one figure.
    """
    mode = _normalize_aggregate(aggregate)

    histories = {seed: _get_history(item) for seed, item in results.items()}
    if not histories:
        print("plot_seeds_overlay: empty `results`")
        return
    n_seeds = len(histories)
    colors = {'train': '#636EFA', 'test': '#EF553B', 'l2_norm': '#FF6692'}

    # Optional l2_norm row
    l2_histories: dict = {}
    if include_l2_norm:
        l2_histories = {s: h for s, h in histories.items() if h.get('l2_norm')}
        if not l2_histories:
            include_l2_norm = False   # silently skip

    # Pre-compute aggregates for loss/acc
    epochs = next(iter(histories.values()))['epoch']
    stacks, aggs, mins, maxs = {}, {}, {}, {}
    for key in ('train_loss', 'test_loss', 'train_acc', 'test_acc'):
        stacked = np.array([h[key] for h in histories.values()])
        stacks[key] = stacked
        aggs[key]   = np.median(stacked, axis=0) if mode == 'min_max' \
                      else (np.mean(stacked, axis=0) if mode == 'mean'
                            else np.median(stacked, axis=0))
        mins[key]   = np.min(stacked, axis=0)
        maxs[key]   = np.max(stacked, axis=0)

    # ... and for l2_norm if needed
    if include_l2_norm:
        l2_epochs   = next(iter(l2_histories.values()))['epoch']
        l2_stacked  = np.array([h['l2_norm'] for h in l2_histories.values()])
        l2_agg      = np.median(l2_stacked, axis=0) if mode == 'min_max' \
                      else (np.mean(l2_stacked, axis=0) if mode == 'mean'
                            else np.median(l2_stacked, axis=0))
        l2_min      = np.min(l2_stacked, axis=0)
        l2_max      = np.max(l2_stacked, axis=0)

    # ----- Build the subplot grid -----
    n_rows = 3 if include_l2_norm else 2
    subplot_titles = ["Loss (linear x)", "Loss (log x)",
                      "Accuracy (linear x)", "Accuracy (log x)"]
    if include_l2_norm:
        subplot_titles += ["||W||² (linear x)", "||W||² (log x)"]

    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
    )

    # ----- Helper to plot one metric on a single panel -----
    def _add_metric_to_panel(row, col, key, name, color,
                              metric_epochs, metric_seeds_dict,
                              metric_agg, metric_min, metric_max,
                              legend_target_row, legend_target_col):
        """Add traces for one metric (train or test, or l2_norm) on (row, col)."""
        show_legend = (row == legend_target_row and col == legend_target_col)

        if mode == 'min_max':
            band_color = _hex_to_rgba(color, 0.2)
            band_x = list(metric_epochs) + list(metric_epochs[::-1])
            band_y = list(metric_max) + list(metric_min[::-1])
            fig.add_trace(
                go.Scatter(
                    x=band_x, y=band_y, fill='toself',
                    fillcolor=band_color, line=dict(width=0),
                    name=f"{name} envelope", legendgroup=name,
                    showlegend=False, hoverinfo='skip',
                ),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=metric_epochs, y=metric_agg, mode='lines',
                    name=f"{name} (median + min/max)",
                    line=dict(color=color, width=agg_width),
                    legendgroup=name, showlegend=show_legend,
                ),
                row=row, col=col,
            )
        else:
            # median or mean: individual seeds + bold aggregate
            for seed, h in metric_seeds_dict.items():
                fig.add_trace(
                    go.Scatter(
                        x=h['epoch'], y=h[key], mode='lines',
                        name=f"{name} seed {seed}",
                        line=dict(color=color),
                        legendgroup=name, showlegend=False,
                        opacity=alpha,
                    ),
                    row=row, col=col,
                )
            fig.add_trace(
                go.Scatter(
                    x=metric_epochs, y=metric_agg, mode='lines',
                    name=f"{name} ({mode})",
                    line=dict(color=color, width=agg_width),
                    legendgroup=name, showlegend=show_legend,
                ),
                row=row, col=col,
            )

    # ----- Row 1: Loss (train + test on each panel) -----
    for col_idx, x_type in [(1, "linear"), (2, "log")]:
        for key, name in [('train_loss', 'train'), ('test_loss', 'test')]:
            _add_metric_to_panel(
                row=1, col=col_idx, key=key, name=name, color=colors[name],
                metric_epochs=epochs, metric_seeds_dict=histories,
                metric_agg=aggs[key], metric_min=mins[key], metric_max=maxs[key],
                legend_target_row=1, legend_target_col=1,
            )
        x_label = "epoch" if col_idx == 1 else "epoch (log)"
        fig.update_xaxes(type=x_type, title_text=x_label, row=1, col=col_idx)
        fig.update_yaxes(type='log', title_text='cross entropy', row=1, col=col_idx)

    # ----- Row 2: Accuracy (train + test on each panel) -----
    for col_idx, x_type in [(1, "linear"), (2, "log")]:
        for key, name in [('train_acc', 'train'), ('test_acc', 'test')]:
            _add_metric_to_panel(
                row=2, col=col_idx, key=key, name=name, color=colors[name],
                metric_epochs=epochs, metric_seeds_dict=histories,
                metric_agg=aggs[key], metric_min=mins[key], metric_max=maxs[key],
                legend_target_row=1, legend_target_col=1,   # legend already shown on row 1
            )
        x_label = "epoch" if col_idx == 1 else "epoch (log)"
        fig.update_xaxes(type=x_type, title_text=x_label, row=2, col=col_idx)
        fig.update_yaxes(range=[0, 1.02], title_text='accuracy', row=2, col=col_idx)

    # ----- Row 3: ||W||² (single metric, shown on each panel) -----
    if include_l2_norm:
        for col_idx, x_type in [(1, "linear"), (2, "log")]:
            _add_metric_to_panel(
                row=3, col=col_idx, key='l2_norm', name='||W||²',
                color=colors['l2_norm'],
                metric_epochs=l2_epochs, metric_seeds_dict=l2_histories,
                metric_agg=l2_agg, metric_min=l2_min, metric_max=l2_max,
                legend_target_row=3, legend_target_col=1,
            )
            x_label = "epoch" if col_idx == 1 else "epoch (log)"
            fig.update_xaxes(type=x_type, title_text=x_label, row=3, col=col_idx)
            fig.update_yaxes(title_text="||W||²", row=3, col=col_idx)

    # ----- Final styling -----
    agg_label = "median + min/max envelope" if mode == 'min_max' \
                else f"bold = {mode}"
    height = 360 * n_rows
    fig.update_layout(
        title=f"{title}  (n={n_seeds} seeds, {agg_label})",
        template='plotly_white', width=1000, height=height,
    )
    fig.show()


def plot_seeds_l2_norm(results: dict, title: str = "||W||² trajectory",
                        alpha: float = 0.4, agg_width: int = 3,
                        aggregate: str = 'median') -> None:
    """Plot the sum of squared weights (||W||²) over training, multi-seed.
    """
    histories = {seed: _get_history(item) for seed, item in results.items()}
    if not histories:
        print("plot_seeds_l2_norm: empty `results`")
        return

    # Filter out seeds with no l2_norm tracked (legacy saved runs)
    have_l2 = {s: h for s, h in histories.items() if h.get('l2_norm')}
    if len(have_l2) < len(histories):
        missing = sorted(set(histories) - set(have_l2))
        print(f"plot_seeds_l2_norm: seeds without l2_norm data skipped: {missing}")

    histories = have_l2
    n_seeds = len(histories)
    color = '#FF6692'   # pink, distinct from train/test
    agg_fn = _aggregate_fn(aggregate)

    epochs = next(iter(histories.values()))['epoch']
    stacked = np.array([h['l2_norm'] for h in histories.values()])
    agg = agg_fn(stacked, axis=0)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("||W||² (linear x)", "||W||² (log x)"),
        horizontal_spacing=0.12,
    )

    for col_idx, x_type in [(1, "linear"), (2, "log")]:
        # 1. Per-seed semi-transparent curves
        for seed, history in histories.items():
            fig.add_trace(
                go.Scatter(
                    x=history['epoch'], y=history['l2_norm'], mode='lines',
                    name=f"seed {seed}",
                    line=dict(color=color),
                    legendgroup='l2_norm',
                    showlegend=False,
                    opacity=alpha,
                ),
                row=1, col=col_idx,
            )
        # 2. Bold aggregate on top
        show_legend = (col_idx == 1)
        fig.add_trace(
            go.Scatter(
                x=epochs, y=agg, mode='lines',
                name=f"||W||² ({aggregate})",
                line=dict(color=color, width=agg_width),
                legendgroup='l2_norm',
                showlegend=show_legend,
            ),
            row=1, col=col_idx,
        )

        x_label = "epoch" if col_idx == 1 else "epoch (log)"
        fig.update_xaxes(type=x_type, title_text=x_label, row=1, col=col_idx)
        fig.update_yaxes(title_text="||W||²  (sum of squared params)",
                          row=1, col=col_idx)

    fig.update_layout(
        title=f"{title}  (n={n_seeds} seeds, bold = {aggregate})",
        template='plotly_white', width=950, height=400,
    )
    fig.show()


def plot_seeds_band(results: dict, title: str = "Multi-seed band",
                     agg_width: int = 3, include_l2_norm: bool = True) -> None:
    plot_seeds_overlay(
        results, title=title,
        agg_width=agg_width,
        aggregate='min_max',
        include_l2_norm=include_l2_norm,
    )


# =============================================================================
#  Nanda-style Fourier component bar chart (W_E + W_U)
# =============================================================================

def plot_fourier_components(
    save_root,
    seeds,
    config,
    *,
    title: str = "Fourier components",
    save_path=None,
    figsize: tuple = (16, 5),
    color_cos: str = "#ff7f0e",
    color_sin: str = "#1f77b4",
    show_seeds: bool = True,
):
    """Nanda-style bar chart: Fourier components of W_E (embedding) and W_U (neuron-logit map).

    Loads `seedN/model.pt` for every `seed in seeds`, projects W_E and W_U onto the
    real Fourier basis over Z_p, and plots per-frequency norms of the cos and sin
    components. Multi-seed aggregation:
        - bar height = mean over seeds
        - error bars = min-max envelope over seeds
        - scatter dots = individual seeds (toggle with `show_seeds`)

    Args
    ----
    save_root : Path or str — directory containing seed{N}/model.pt subfolders
    seeds     : iterable of int — seeds to load and aggregate
    config    : Config object — used to build the Fourier basis (needs .p, .device)

    Returns
    -------
    dict : {seed: {'we_cos','we_sin','wu_cos','wu_sin'}} of np.ndarray (length p//2)
    """
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt
    from fourier_metrics import make_fourier_basis

    save_root = Path(save_root)
    p = config.p

    basis    = make_fourier_basis(config).cpu()
    cos_rows = basis[1::2]    # (p//2, p)  → cos(1), cos(2), ...
    sin_rows = basis[2::2]    # (p//2, p)  → sin(1), sin(2), ...
    n_freqs  = cos_rows.shape[0]
    freqs    = np.arange(1, n_freqs + 1)

    def _find_param(state, suffix):
        for k, v in state.items():
            if k.endswith(suffix):
                return v
        return None

    per_seed = {}
    for s in seeds:
        mp = save_root / f"seed{s}" / "model.pt"
        if not mp.exists():
            print(f"  plot_fourier_components: skip seed {s} → {mp} not found")
            continue
        state = t.load(mp, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        W_E = _find_param(state, "W_E")
        W_U = _find_param(state, "W_U")
        if W_E is None or W_U is None:
            raise KeyError(f"W_E or W_U not found in {mp}. Keys: {list(state)}")

        W_E_p = W_E[:, :p].float().cpu()    # (d_model, p)
        W_U_p = W_U[:, :p].float().cpu()    # (d_model, p)
        per_seed[s] = {
            "we_cos": (W_E_p @ cos_rows.T).norm(dim=0).numpy(),
            "we_sin": (W_E_p @ sin_rows.T).norm(dim=0).numpy(),
            "wu_cos": (W_U_p @ cos_rows.T).norm(dim=0).numpy(),
            "wu_sin": (W_U_p @ sin_rows.T).norm(dim=0).numpy(),
        }

    if not per_seed:
        print(f"plot_fourier_components: no seeds loaded from {save_root}")
        return None

    def _agg(key):
        stk = np.array([per_seed[s][key] for s in per_seed])    # (n_seeds, n_freqs)
        return stk.mean(0), stk.min(0), stk.max(0), stk

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panels = [
        ("we_cos", "we_sin", "Fourier Components of Embedding Matrix ($W_E$)"),
        ("wu_cos", "wu_sin", "Fourier Components of Neuron-Logit Map ($W_U$)"),
    ]
    width = 0.42
    for ax, (key_c, key_s, panel_title) in zip(axes, panels):
        c_mean, c_min, c_max, c_stk = _agg(key_c)
        s_mean, s_min, s_max, s_stk = _agg(key_s)
        # bars : mean + min/max errbar
        ax.bar(freqs - width/2, c_mean, width,
               yerr=[c_mean - c_min, c_max - c_mean],
               color=color_cos, alpha=0.85, edgecolor="black", linewidth=0.4,
               label="cos", capsize=0, error_kw={"lw": 0.7, "alpha": 0.6})
        ax.bar(freqs + width/2, s_mean, width,
               yerr=[s_mean - s_min, s_max - s_mean],
               color=color_sin, alpha=0.85, edgecolor="black", linewidth=0.4,
               label="sin", capsize=0, error_kw={"lw": 0.7, "alpha": 0.6})
        # per-seed dots
        if show_seeds:
            for i in range(c_stk.shape[0]):
                ax.scatter(freqs - width/2, c_stk[i], color=color_cos, alpha=0.45, s=8, zorder=4)
                ax.scatter(freqs + width/2, s_stk[i], color=color_sin, alpha=0.45, s=8, zorder=4)
        ax.set_xlabel("Frequency k")
        ax.set_ylabel("Norm of Fourier Component")
        ax.set_title(panel_title, fontsize=11)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
        ax.grid(True, alpha=0.3, axis="y", linestyle=":")

    plt.suptitle(f"{title}   ·   n={len(per_seed)} seeds   ·   bar = mean, errbar = min-max, dots = per-seed",
                 y=1.02, fontsize=12, fontweight="600")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.show()
    return per_seed


# =============================================================================
#  Multi-seed Fourier loss comparison (matplotlib + band style)
# =============================================================================

def plot_fourier_loss_multiseed(
    runs_dict,
    *,
    title: str = "Fourier progress losses",
    excluded_key: str = "excluded_all_loss_train",
    restricted_key: str = "restricted_loss_test",
    mem_med=None, circuit_med=None, grok_med=None,
    save_path=None,
    figsize: tuple = (15, 5.5),
    log_scale: bool = True,
):
    """Multi-seed matplotlib version of `plot_fourier_loss_comparison`.

    Shows 2 panels (train/test/excluded, train/test/restricted). For each curve:
    bold mean across seeds + transparent per-seed + min-max envelope.
    Optionally adds vertical markers for memorization / circuit formation / grokking.

    runs_dict : {seed: history_dict}
    """
    import matplotlib.pyplot as plt
    if not runs_dict:
        print("plot_fourier_loss_multiseed: empty runs_dict"); return

    histories = list(runs_dict.values())
    if not any(h.get("fourier_epoch") for h in histories):
        print("plot_fourier_loss_multiseed: no Fourier snapshots"); return

    def _stack(hist_list, key, time_key="epoch"):
        arrays, eps = [], None
        for h in hist_list:
            if time_key not in h or key not in h: continue
            ep, val = np.array(h[time_key]), np.array(h[key])
            if eps is None: eps = ep
            else:
                n = min(len(eps), len(ep))
                eps = eps[:n]; arrays = [a[:n] for a in arrays]; val = val[:n]
            arrays.append(val)
        return (eps, np.stack(arrays, axis=0)) if arrays else (None, None)

    def _plot_band(ax, eps, stk, color, label, alpha_seed=0.25, alpha_band=0.15):
        if stk is None or stk.size == 0: return
        ax.fill_between(eps, stk.min(0), stk.max(0), color=color, alpha=alpha_band, zorder=2)
        for i in range(stk.shape[0]):
            ax.plot(eps, stk[i], color=color, lw=0.7, alpha=alpha_seed, zorder=3)
        ax.plot(eps, stk.mean(0), color=color, lw=2.6, label=label, zorder=4)

    def _vlines(ax):
        if mem_med:     ax.axvline(mem_med,     color="#1f77b4", ls=":",  lw=1.5, alpha=0.75, label=f"mem = {int(mem_med)}")
        if circuit_med: ax.axvline(circuit_med, color="#9467bd", ls="--", lw=1.5, alpha=0.75, label=f"circuit = {int(circuit_med)}")
        if grok_med:    ax.axvline(grok_med,    color="#cc1f1a", ls="-.", lw=1.5, alpha=0.75, label=f"grok = {int(grok_med)}")

    ep_l, train_loss = _stack(histories, "train_loss")
    _,    test_loss  = _stack(histories, "test_loss")
    ep_f, excl       = _stack(histories, excluded_key,  time_key="fourier_epoch")
    _,    restr      = _stack(histories, restricted_key, time_key="fourier_epoch")

    COL_TRAIN, COL_TEST = "#1f77b4", "#d62728"
    COL_EXC,   COL_RES  = "#2ca02c", "#9467bd"

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # Panel 1 : train/test + excluded
    ax = axes[0]
    _plot_band(ax, ep_l, train_loss, COL_TRAIN, "train loss")
    _plot_band(ax, ep_l, test_loss,  COL_TEST,  "test loss")
    if excl is not None:
        _plot_band(ax, ep_f, excl, COL_EXC, f"{excluded_key} (key freqs removed)")
    _vlines(ax)
    if log_scale: ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross entropy")
    ax.set_title(f"Train/test + excluded")
    ax.legend(loc="best", fontsize=8); ax.grid(True, which="both", alpha=0.3, linestyle=":")

    # Panel 2 : train/test + restricted
    ax = axes[1]
    _plot_band(ax, ep_l, train_loss, COL_TRAIN, "train loss")
    _plot_band(ax, ep_l, test_loss,  COL_TEST,  "test loss")
    if restr is not None:
        _plot_band(ax, ep_f, restr, COL_RES, f"{restricted_key} (key freqs only)")
    _vlines(ax)
    if log_scale: ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross entropy")
    ax.set_title(f"Train/test + restricted")
    ax.legend(loc="best", fontsize=8); ax.grid(True, which="both", alpha=0.3, linestyle=":")

    plt.suptitle(f"{title}  ·  n={len(runs_dict)} seeds  ·  bold = mean, transparent = per-seed, band = min–max",
                 y=1.02, fontsize=12, fontweight="600")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.show()
