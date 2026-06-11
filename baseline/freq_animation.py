"""
freq_animation.py
==================

Standalone module for animating the frequency mass landscape over training.
This file is independent from viz_analysis.py — you can include/exclude/move it freely.

2 backends provided :
    - `animate_freq_mass_plotly` : interactive HTML (slider + play/pause), saved as .html
    - `animate_freq_mass_mpl`    : MP4 or GIF via matplotlib FuncAnimation

Usage example :
    from freq_animation import animate_freq_mass_plotly, animate_freq_mass_mpl
    animate_freq_mass_plotly(runs, save_path='nanda_anim.html')   # interactive
    animate_freq_mass_mpl(runs, save_path='nanda_anim.mp4')       # video
"""

from __future__ import annotations
from pathlib import Path

import numpy as np


# =============================================================================
# Internal helper : robust multi-seed stacking (handles adaptive_logging)
# =============================================================================
def _stack_freq_matrix(runs, key='wl_frequency_masses'):
    """Stack per-seed (n_snapshots × n_freqs) Fourier matrices on a common epoch grid
    via interpolation. Returns (epochs, mean_matrix). Mean is nan-aware to handle seeds
    with shorter histories (e.g. early-grok seeds with adaptive_logging)."""
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
    all_epochs = np.array(sorted(set().union(*[set(ep.tolist()) for ep in raw_eps])))
    if all_epochs.size == 0:
        return None, None
    n_freqs = raw_mats[0].shape[1]
    interp = np.full((len(raw_mats), len(all_epochs), n_freqs), np.nan)
    for i, (ep, m) in enumerate(zip(raw_eps, raw_mats)):
        for j in range(n_freqs):
            interp[i, :, j] = np.interp(all_epochs, ep, m[:, j],
                                          left=np.nan, right=np.nan)
    with np.errstate(all='ignore'):
        mean_mat = np.nanmean(interp, axis=0)
    return all_epochs, mean_mat


# =============================================================================
# Plotly interactive animation (HTML — slider + play/pause)
# =============================================================================
def animate_freq_mass_plotly(
    runs,
    *,
    key: str = 'wl_frequency_masses',
    title: str = 'Frequency mass emergence — interactive',
    save_path=None,
    log_z: bool = True,
    n_frames: int = 60,
    frame_duration: int = 150,    # ms per frame
    colorscale: str = 'Magma',
    width: int = 1100, height: int = 720,
):
    """3D animated surface plot (Plotly) — the surface "grows" as epochs progress.

    At frame t, shows the (epoch × freq → mass) surface from epoch 0 up to epoch t.
    Interactive : slider to scrub, play/pause buttons, rotate with mouse.

    Args
    ----
    runs       : list of {'seed', 'history', ...} (output of viz_analysis.load_seeds)
    key        : 'wl_frequency_masses' (default) or 'we_frequency_masses'
    save_path  : if set, saves to HTML (.html). Otherwise just shown in notebook.
    n_frames   : number of frames in the animation (downsampled from snapshots)
    log_z      : log10 scale on Z axis (recommended : reveals key freqs)

    Returns the plotly Figure object.
    """
    import plotly.graph_objects as go

    epochs, mat = _stack_freq_matrix(runs, key=key)
    if mat is None:
        print('animate_freq_mass_plotly: no data'); return None

    freqs = np.arange(1, mat.shape[1] + 1)
    if log_z:
        Z_full = np.log10(np.maximum(np.nan_to_num(mat, nan=1e-10), 1e-10))
        z_label = 'log10(mass)'
    else:
        Z_full = np.nan_to_num(mat, nan=0.0)
        z_label = 'mass'

    n_epochs = len(epochs)
    frame_idx = np.linspace(0, n_epochs - 1, min(n_frames, n_epochs)).astype(int)
    z_min, z_max = float(np.nanmin(Z_full)), float(np.nanmax(Z_full))
    ep_min, ep_max = float(epochs[0]), float(epochs[-1])

    # ── Frames ────────────────────────────────────────────────────────────────
    frames = []
    for t in frame_idx:
        Z_t  = Z_full[:t+1, :].T          # (n_freqs, t+1)
        ep_t = epochs[:t+1]
        frames.append(go.Frame(
            data=[go.Surface(
                z=Z_t, x=ep_t, y=freqs,
                colorscale=colorscale,
                cmin=z_min, cmax=z_max,
                showscale=True,
                colorbar=dict(title=z_label, len=0.8),
                hovertemplate=('epoch=%{x:.0f}<br>freq k=%{y}<br>'
                                + z_label + '=%{z:.3f}<extra></extra>'),
            )],
            name=f'epoch_{int(epochs[t])}',
        ))

    # ── Initial trace (first frame) ───────────────────────────────────────────
    init_trace = go.Surface(
        z=Z_full[:1, :].T, x=epochs[:1], y=freqs,
        colorscale=colorscale, cmin=z_min, cmax=z_max,
        showscale=True, colorbar=dict(title=z_label, len=0.8),
    )

    fig = go.Figure(data=[init_trace], frames=frames)

    # ── Slider + play/pause buttons ───────────────────────────────────────────
    sliders = [dict(
        active=0, pad=dict(t=50), len=0.85, x=0.08, y=0,
        currentvalue=dict(prefix='epoch = ', font=dict(size=14)),
        steps=[dict(
            method='animate',
            args=[[f'epoch_{int(epochs[t])}'],
                  dict(mode='immediate',
                       frame=dict(duration=frame_duration, redraw=True),
                       transition=dict(duration=50))],
            label=str(int(epochs[t])),
        ) for t in frame_idx],
    )]
    updatemenus = [dict(
        type='buttons', showactive=False,
        x=0.04, y=-0.05,
        buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, dict(frame=dict(duration=frame_duration, redraw=True),
                                  fromcurrent=True,
                                  transition=dict(duration=50))]),
            dict(label='⏸ Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode='immediate',
                                    transition=dict(duration=0))]),
        ],
    )]

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        scene=dict(
            xaxis=dict(title='epoch', range=[ep_min, ep_max]),
            yaxis=dict(title='frequency index k', range=[1, mat.shape[1]]),
            zaxis=dict(title=z_label, range=[z_min, z_max]),
            camera=dict(eye=dict(x=1.6, y=1.5, z=1.0)),
            aspectratio=dict(x=1.5, y=1.0, z=0.8),
        ),
        sliders=sliders, updatemenus=updatemenus,
        width=width, height=height,
        margin=dict(l=0, r=0, t=50, b=70),
    )

    if save_path is not None:
        save_path = str(save_path)
        if not save_path.endswith('.html'):
            save_path = save_path.rsplit('.', 1)[0] + '.html'
        fig.write_html(save_path)
        print(f'  ✓ saved animated 3D to {save_path}')

    fig.show()
    return fig


# =============================================================================
# Matplotlib animation (MP4 / GIF)
# =============================================================================
def animate_freq_mass_mpl(
    runs,
    *,
    key: str = 'wl_frequency_masses',
    title: str = 'Frequency mass emergence',
    save_path='freq_animation.gif',     # ← default GIF (no ffmpeg needed)
    log_z: bool = True,
    n_frames: int = 50,                  # ← reduced (3D render is slow)
    fps: int = 10, dpi: int = 90,
    cmap: str = 'magma',
    elev: float = 30, azim: float = -60,
    figsize: tuple = (10, 6),
):
    """3D animated surface plot (matplotlib) — saved as MP4 or GIF.

    Auto-detects ffmpeg : if `save_path` is .mp4 but ffmpeg is missing, falls back
    to .gif automatically. Prints progress every 10% of frames.

    Note : 3D surface animation is SLOW (~1-3s per frame). Use small n_frames
    (50-80) for quick test, larger for final. Plotly version is much faster
    and interactive.

    Args :
        n_frames  : number of frames (50 default is fast, ~30s total)
        fps       : frames per second of output video
        dpi       : resolution (90 default; bump to 120+ for HD)
    """
    import shutil
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    epochs, mat = _stack_freq_matrix(runs, key=key)
    if mat is None:
        print('animate_freq_mass_mpl: no data'); return None

    # ── Detect writer + auto-correct extension ────────────────────────────────
    save_path = str(save_path)
    ext = save_path.rsplit('.', 1)[-1].lower()
    has_ffmpeg = shutil.which('ffmpeg') is not None
    if ext == 'mp4':
        if not has_ffmpeg:
            new_path = save_path.rsplit('.', 1)[0] + '.gif'
            print(f'  ⚠ ffmpeg not found in PATH — saving as GIF instead :')
            print(f'     {save_path}  →  {new_path}')
            save_path = new_path
            writer_name = 'pillow'
        else:
            writer_name = 'ffmpeg'
    elif ext == 'gif':
        writer_name = 'pillow'
    else:
        print(f'  ⚠ unknown extension .{ext} — defaulting to .gif')
        save_path = save_path.rsplit('.', 1)[0] + '.gif'
        writer_name = 'pillow'

    freqs = np.arange(1, mat.shape[1] + 1)
    if log_z:
        Z_full = np.log10(np.maximum(np.nan_to_num(mat, nan=1e-10), 1e-10))
        z_label = 'log10(mass)'
    else:
        Z_full = np.nan_to_num(mat, nan=0.0)
        z_label = 'mass'

    n_epochs = len(epochs)
    frame_idx = np.linspace(0, n_epochs - 1, min(n_frames, n_epochs)).astype(int)
    z_min, z_max = float(np.nanmin(Z_full)), float(np.nanmax(Z_full))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    def render(t_idx):
        ax.clear()
        Z_t  = Z_full[:t_idx+1, :].T
        ep_t = epochs[:t_idx+1]
        X, Y = np.meshgrid(ep_t, freqs)
        ax.plot_surface(X, Y, Z_t, cmap=cmap, edgecolor='none',
                         alpha=0.92, vmin=z_min, vmax=z_max,
                         rstride=1, cstride=1)
        ax.set_xlim(epochs[0], epochs[-1])
        ax.set_ylim(1, mat.shape[1])
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('epoch'); ax.set_ylabel('frequency k')
        ax.set_zlabel(z_label)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'{title}\nepoch = {int(epochs[t_idx])}', fontsize=11)
        return [ax]

    anim = FuncAnimation(fig, render, frames=frame_idx, blit=False, repeat=False)

    # ── Progress callback ─────────────────────────────────────────────────────
    n_total = len(frame_idx)
    step = max(1, n_total // 10)
    def progress_cb(i, n):
        if i % step == 0 or i == n_total - 1:
            print(f'    frame {i+1}/{n_total} ({100*(i+1)/n_total:.0f}%)')

    print(f'  Rendering {n_total} frames ({writer_name}) → {save_path} ...')
    try:
        anim.save(save_path, writer=writer_name, fps=fps, dpi=dpi,
                  progress_callback=progress_cb)
        print(f'  ✓ saved to {save_path}')
    except KeyboardInterrupt:
        print(f'\n  ⏸ interrupted — partial file may exist at {save_path}')
        raise
    finally:
        plt.close(fig)
    return anim
