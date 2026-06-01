from __future__ import annotations
import numpy as np
import torch as t
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#!REPRIS LES FONCTIONS DE JEAN DANS BASELINE.IPYNB MAIS ADAPTE A LA NOUVELLE PIPELINE

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


def plot_seeds_overlay(results: dict, title: str = "Multi-seed overlay",
                        alpha: float = 0.4) -> None:
    """Plot all seeds as semi-transparent curves on a 2x2 grid.
    """
    histories = {seed: _get_history(item) for seed, item in results.items()}
    if not histories:
        print("plot_seeds_overlay: empty `results`")
        return
    n_seeds = len(histories)
    colors = {'train': '#636EFA', 'test': '#EF553B'}

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Loss (linear x)", "Accuracy (linear x)",
            "Loss (log x)",    "Accuracy (log x)",
        ),
        vertical_spacing=0.15,
    )

    # For each col, the (metric_key, legend_name) pairs to plot.
    col_metrics = {
        1: [('train_loss', 'train'), ('test_loss', 'test')],
        2: [('train_acc',  'train'), ('test_acc',  'test')],
    }

    for row_idx, x_type in [(1, "linear"), (2, "log")]:
        for col_idx in (1, 2):
            for key, name in col_metrics[col_idx]:
                color = colors[name]
                for seed_idx, (seed, history) in enumerate(histories.items()):
                    # Show legend ONCE per metric type
                    show_legend = (row_idx == 1 and col_idx == 1 and seed_idx == 0)
                    fig.add_trace(
                        go.Scatter(
                            x=history['epoch'], y=history[key], mode='lines',
                            name=name, line=dict(color=color),
                            legendgroup=name,
                            showlegend=show_legend,
                            opacity=alpha,
                            hovertext=f"seed {seed}",
                        ),
                        row=row_idx, col=col_idx,
                    )

        # Axes for this row
        x_label = "epoch" if row_idx == 1 else "epoch (log)"
        fig.update_xaxes(type=x_type, title_text=x_label, row=row_idx, col=1)
        fig.update_xaxes(type=x_type, title_text=x_label, row=row_idx, col=2)
        fig.update_yaxes(type='log', title_text='cross entropy', row=row_idx, col=1)
        fig.update_yaxes(range=[0, 1.02], title_text='accuracy', row=row_idx, col=2)

    fig.update_layout(
        title=f"{title}  (n={n_seeds} seeds)",
        template='plotly_white', width=950, height=720,
    )
    fig.show()


def plot_seeds_band(results: dict, title: str = "Multi-seed band") -> None:
    """Plot median (solid line) + min/max envelope (band) on a 2x2 grid.
    """
    histories = [_get_history(item) for item in results.values()]
    if not histories:
        print("plot_seeds_band: empty `results`")
        return
    epochs = histories[0]['epoch']
    n_seeds = len(histories)

    colors = {'train': '#636EFA', 'test': '#EF553B'}

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Loss (linear x)", "Accuracy (linear x)",
            "Loss (log x)",    "Accuracy (log x)",
        ),
        vertical_spacing=0.15,
    )

    # Pre-compute median + min + max for all 4 metrics
    def _stats(key):
        stacked = np.array([h[key] for h in histories])      # (n_seeds, n_epochs)
        return np.median(stacked, axis=0), np.min(stacked, axis=0), np.max(stacked, axis=0)

    stats = {
        'train_loss': _stats('train_loss'),
        'test_loss':  _stats('test_loss'),
        'train_acc':  _stats('train_acc'),
        'test_acc':   _stats('test_acc'),
    }

    band_x = list(epochs) + list(epochs[::-1])

    # For each col, the (metric_key, legend_name) pairs to plot.
    # Col 1 = Loss panel (train_loss + test_loss).
    # Col 2 = Accuracy panel (train_acc + test_acc).
    col_metrics = {
        1: [('train_loss', 'train'), ('test_loss', 'test')],
        2: [('train_acc',  'train'), ('test_acc',  'test')],
    }

    for row_idx, x_type in [(1, "linear"), (2, "log")]:
        for col_idx in (1, 2):
            for key, name in col_metrics[col_idx]:
                median, vmin, vmax = stats[key]
                color = colors[name]
                band_color = _hex_to_rgba(color, 0.2)
                band_y = list(vmax) + list(vmin[::-1])

                # Envelope (band) — no legend
                fig.add_trace(
                    go.Scatter(
                        x=band_x, y=band_y, fill='toself',
                        fillcolor=band_color, line=dict(width=0),
                        name=f"{name} envelope", legendgroup=name,
                        showlegend=False, hoverinfo='skip',
                    ),
                    row=row_idx, col=col_idx,
                )
                # Median line
                show_legend = (row_idx == 1 and col_idx == 1)   # one legend entry per name
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=median, mode='lines',
                        line=dict(color=color, width=2),
                        name=name, legendgroup=name,
                        showlegend=show_legend,
                    ),
                    row=row_idx, col=col_idx,
                )

        # Axes for this row
        x_label = "epoch" if row_idx == 1 else "epoch (log)"
        fig.update_xaxes(type=x_type, title_text=x_label, row=row_idx, col=1)
        fig.update_xaxes(type=x_type, title_text=x_label, row=row_idx, col=2)
        fig.update_yaxes(type='log', title_text='cross entropy', row=row_idx, col=1)
        fig.update_yaxes(range=[0, 1.02], title_text='accuracy', row=row_idx, col=2)

    fig.update_layout(
        title=f"{title}  (n={n_seeds} seeds, median + min/max envelope)",
        template='plotly_white', width=950, height=720,
    )
    fig.show()
