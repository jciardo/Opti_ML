"""
Nanda-style Fourier metric measures for the modular-addition toy transformer.
Reference: Nanda et. al(2023),
"Progress measures for grokking via mechanistic interpretability"
(https://arxiv.org/abs/2301.05217).

Reference implementation: the Grokking_Analysis.ipynb notebook from
https://github.com/mechanistic-interpretability/progress-measures-paper —

"""

from __future__ import annotations
import math
from typing import Iterable, Optional
import einops
import torch as t
import torch.nn.functional as F


def make_fourier_basis(config) -> t.Tensor:
    """Real orthonormal Fourier basis over Z_p.

    Row layout matches Nanda's code:
      0: constant, 2*k-1: cos(k), 2*k: sin(k)
    for k = 1, ..., p//2.
    """
    p = config.p
    device = config.device
    arange = t.arange(p, device=device)
    basis = [t.ones(p, device=device) / math.sqrt(p)]
    for k in range(1, p // 2 + 1):
        cos_k = t.cos(2 * t.pi * arange * k / p)
        sin_k = t.sin(2 * t.pi * arange * k / p)
        basis.append(cos_k / cos_k.norm())
        basis.append(sin_k / sin_k.norm())
    return t.stack(basis, dim=0)


def get_train_test_masks(
    train_pairs: Iterable[tuple[int, int, int]],
    test_pairs: Iterable[tuple[int, int, int]],
    p: int,
    *,
    device,
) -> tuple[t.Tensor, t.Tensor]:
    train_set = {(int(a), int(b)) for a, b, _ in train_pairs}
    test_set = {(int(a), int(b)) for a, b, _ in test_pairs}
    is_train = []
    is_test = []
    for a in range(p):
        for b in range(p):
            is_train.append((a, b) in train_set)
            is_test.append((a, b) in test_set)
    return (
        t.tensor(is_train, dtype=t.bool, device=device),
        t.tensor(is_test, dtype=t.bool, device=device),
    )


def labels_for_all_data(config, all_data: t.Tensor) -> t.Tensor:
    # Current baseline task is modular addition.
    return (all_data[:, 0] + all_data[:, 1]) % config.p


def cross_entropy_high_precision(logits: t.Tensor, labels: t.Tensor) -> t.Tensor:
    target_dtype = t.float32 if logits.device.type == "mps" else t.float64
    logprobs = F.log_softmax(logits.to(target_dtype), dim=-1)
    return -t.gather(logprobs, index=labels[:, None], dim=-1).mean()


def _as_full_grid_logits(logits: t.Tensor, p: int) -> t.Tensor:
    if logits.shape[1] == p * p:
        logits = logits.T
    if logits.shape == t.Size([p * p, p + 1]):
        logits = logits[:, :p]
    return logits.reshape(p * p, p)


def evaluate_logits(
    logits: t.Tensor,
    *,
    p: int,
    labels: t.Tensor,
    is_train: t.Tensor,
    is_test: t.Tensor,
    mode: str = "all",
    bias_correction: bool = False,
    original_logits: Optional[t.Tensor] = None,
) -> tuple[float, float]:
    logits = _as_full_grid_logits(logits, p)
    if bias_correction:
        if original_logits is None:
            raise ValueError("original_logits is required for bias_correction=True")
        original_logits = _as_full_grid_logits(original_logits, p)
        logits = einops.reduce(original_logits - logits, "batch cls -> cls", "mean") + logits

    if mode == "train":
        idx = is_train
    elif mode == "test":
        idx = is_test
    elif mode == "all":
        idx = slice(None)
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected train, test, or all.")

    selected_logits = logits[idx]
    selected_labels = labels[idx]
    loss = cross_entropy_high_precision(selected_logits, selected_labels)
    acc = (selected_logits.argmax(dim=-1) == selected_labels).float().mean()
    return loss.item(), acc.item()


def fourier_2d_basis_term(x_index: int, y_index: int, fourier_basis: t.Tensor) -> t.Tensor:
    return (fourier_basis[x_index][:, None] * fourier_basis[y_index][None, :]).flatten()


def get_component_cos_xpy(
    tensor: t.Tensor,
    freq: int,
    fourier_basis: t.Tensor,
    *,
    collapse_dim: bool = False,
) -> t.Tensor:
    cosx_cosy = fourier_2d_basis_term(2 * freq - 1, 2 * freq - 1, fourier_basis)
    sinx_siny = fourier_2d_basis_term(2 * freq, 2 * freq, fourier_basis)
    direction = (cosx_cosy - sinx_siny) / math.sqrt(2)
    if collapse_dim:
        return direction @ tensor
    return direction[:, None] @ (direction[None, :] @ tensor)


def get_component_sin_xpy(
    tensor: t.Tensor,
    freq: int,
    fourier_basis: t.Tensor,
    *,
    collapse_dim: bool = False,
) -> t.Tensor:
    sinx_cosy = fourier_2d_basis_term(2 * freq, 2 * freq - 1, fourier_basis)
    cosx_siny = fourier_2d_basis_term(2 * freq - 1, 2 * freq, fourier_basis)
    direction = (sinx_cosy + cosx_siny) / math.sqrt(2)
    if collapse_dim:
        return direction @ tensor
    return direction[:, None] @ (direction[None, :] @ tensor)


def trig_component(logits: t.Tensor, freq: int, fourier_basis: t.Tensor) -> t.Tensor:
    return (
        get_component_cos_xpy(logits, freq, fourier_basis)
        + get_component_sin_xpy(logits, freq, fourier_basis)
    )


def _pair_frequency_masses(fourier_matrix: t.Tensor, p: int) -> t.Tensor:
    masses = []
    for k in range(1, p // 2 + 1):
        masses.append(
            fourier_matrix[2 * k - 1].pow(2).sum()
            + fourier_matrix[2 * k].pow(2).sum()
        )
    return t.stack(masses)


def wl_frequency_masses(model, config, fourier_basis: t.Tensor) -> t.Tensor:
    p = config.p
    # W_L maps MLP post-activation neurons directly to logits.
    w_u = model.unembed.W_U[:, :p].T
    w_out = model.blocks[0].mlp.W_out
    w_l = w_u @ w_out
    return _pair_frequency_masses(fourier_basis @ w_l, p)


def embedding_frequency_masses(model, config, fourier_basis: t.Tensor) -> t.Tensor:
    p = config.p
    w_e = model.embed.W_E[:, :p]
    fourier_embed = w_e @ fourier_basis.T
    masses = []
    for k in range(1, p // 2 + 1):
        masses.append(
            fourier_embed[:, 2 * k - 1].pow(2).sum()
            + fourier_embed[:, 2 * k].pow(2).sum()
        )
    return t.stack(masses)


def choose_key_freqs_from_wl(
    model,
    config,
    fourier_basis: t.Tensor,
    *,
    top_k: int = 5,
    rel_threshold: Optional[float] = None,
) -> list[int]:
    masses = wl_frequency_masses(model, config, fourier_basis)
    order = t.argsort(masses, descending=True)
    if rel_threshold is not None and masses.numel() > 0:
        keep = masses[order] >= masses[order[0]] * rel_threshold
        order = order[keep]
    order = order[:top_k]
    return sorted((order + 1).detach().cpu().tolist())


def calculate_coefficients(logits: t.Tensor, config) -> t.Tensor:
    p = config.p
    device = config.device
    x = t.arange(p, device=device)[None, :, None, None]
    y = t.arange(p, device=device)[None, None, :, None]
    z = t.arange(p, device=device)[None, None, None, :]
    w = t.arange(1, p // 2 + 1, device=device)[:, None, None, None]
    coses = t.cos(w * t.pi * 2 / p * (x + y - z))
    coses = coses.reshape(p // 2, p * p, p)
    coses = coses / coses.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt()
    return (coses * logits).sum(dim=(-2, -1))


def gini(values: t.Tensor) -> float:
    x = values.detach().flatten().float()
    x = x[x >= 0]
    if x.numel() == 0:
        return 0.0
    total = x.sum()
    if total <= 0:
        return 0.0
    x, _ = x.sort()
    n = x.numel()
    idx = t.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return ((2 * idx - n - 1) * x).sum().div(n * total).item()


def entropy(values: t.Tensor) -> float:
    x = values.detach().flatten().float()
    total = x.sum()
    if total <= 0:
        return 0.0
    probs = x / total
    probs = probs[probs > 0]
    return (-(probs * probs.log()).sum()).item()


def topk_concentration(values: t.Tensor, k: int = 5) -> float:
    x = values.detach().flatten().float()
    total = x.sum()
    if total <= 0:
        return 0.0
    return t.topk(x, min(k, x.numel())).values.sum().div(total).item()


@t.no_grad()
def fourier_metrics(
    model,
    config,
    all_data: t.Tensor,
    is_train: t.Tensor,
    is_test: t.Tensor,
    *,
    key_freqs: Optional[Iterable[int]] = None,
    key_freq_top_k: int = 5,
) -> dict:
    """Compute Nanda-style Fourier progress measures for one checkpoint."""
    p = config.p
    fourier_basis = make_fourier_basis(config)
    labels = labels_for_all_data(config, all_data)
    logits = model(all_data)[:, -1, :p]

    if key_freqs is None:
        key_freqs = choose_key_freqs_from_wl(
            model, config, fourier_basis, top_k=key_freq_top_k
        )
    else:
        key_freqs = sorted(int(k) for k in key_freqs)

    components = [trig_component(logits, freq, fourier_basis) for freq in key_freqs]
    restricted_logits = sum(components) if components else t.zeros_like(logits)
    excluded_logits = logits - restricted_logits

    out = {"key_freqs": key_freqs}
    for mode in ("all", "train", "test"):
        loss, acc = evaluate_logits(
            restricted_logits,
            p=p,
            labels=labels,
            is_train=is_train,
            is_test=is_test,
            mode=mode,
            bias_correction=True,
            original_logits=logits,
        )
        out[f"restricted_loss_{mode}"] = loss
        out[f"restricted_acc_{mode}"] = acc

    for mode in ("all", "train", "test"):
        loss, acc = evaluate_logits(
            excluded_logits,
            p=p,
            labels=labels,
            is_train=is_train,
            is_test=is_test,
            mode=mode,
            bias_correction=False,
        )
        out[f"excluded_all_loss_{mode}"] = loss
        out[f"excluded_all_acc_{mode}"] = acc

    per_freq_excluded = []
    for freq, comp in zip(key_freqs, components):
        loss, _ = evaluate_logits(
            logits - comp,
            p=p,
            labels=labels,
            is_train=is_train,
            is_test=is_test,
            mode="train",
            bias_correction=False,
        )
        per_freq_excluded.append(loss)

    wl_masses = wl_frequency_masses(model, config, fourier_basis)
    we_masses = embedding_frequency_masses(model, config, fourier_basis)
    coefficients = calculate_coefficients(logits, config)

    # Backwards-compatible aliases for the existing pipeline history keys.
    out["restricted_loss"] = out["restricted_loss_all"]
    out["excluded_loss"] = per_freq_excluded
    out["excluded_loss_mean"] = (
        float(sum(per_freq_excluded) / len(per_freq_excluded))
        if per_freq_excluded else 0.0
    )
    out["gini_W_E"] = gini(we_masses)
    out["gini_W_L"] = gini(wl_masses)

    out.update({
        "wl_top5_concentration": topk_concentration(wl_masses, 5),
        "we_top5_concentration": topk_concentration(we_masses, 5),
        "wl_entropy": entropy(wl_masses),
        "we_entropy": entropy(we_masses),
        "wl_frequency_masses": wl_masses.detach().cpu().tolist(),
        "we_frequency_masses": we_masses.detach().cpu().tolist(),
        "cos_coefficients": coefficients.detach().cpu().tolist(),
        "key_cos_coefficients": coefficients[[k - 1 for k in key_freqs]].detach().cpu().tolist()
            if key_freqs else [],
    })

    # Spectral concentration on the fixed key_freqs (only when caller provided them):
    # fraction of the spectral mass carried by the freqs identified in Phase 1.
    if key_freqs:
        key_idx = [k - 1 for k in key_freqs if 1 <= k <= len(wl_masses)]
        if key_idx:
            wl_total = wl_masses.sum()
            we_total = we_masses.sum()
            out["wl_keyfreq_concentration"] = float(wl_masses[key_idx].sum() / wl_total) if wl_total > 0 else 0.0
            out["we_keyfreq_concentration"] = float(we_masses[key_idx].sum() / we_total) if we_total > 0 else 0.0

    return out
