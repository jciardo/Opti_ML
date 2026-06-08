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
