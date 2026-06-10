"""
PyTorch port of Egalitarian Gradient Descent (EGD).

Paper      : Saheb Pasand & Dohmatob, "Egalitarian Gradient Descent:
             A Simple Approach to Accelerated Grokking", ICLR 2026.
             https://arxiv.org/abs/2510.04930
Ref code   : https://github.com/asahebpa/Egalitarian-Gradient-Descent
             (JAX in main-results/, PyTorch fragments in grokfast-comparison/)
Port       : Faithful PyTorch optimizer-class port for plug-in via OptimizerSpec.
             Matches Algorithms 1-3 of Appendix G.

Method (paragraph 4, eq. 11) :
    For each gradient matrix G of shape (m, n) (m = fan-out, n = fan-in),
        G_tilde = (G G^T)^{-1/2} G = U V^T,    with G = U S V^T  (SVD).
    All singular values become 1, while left/right singular vectors of G
    are preserved. Non-2D grads (biases, 1D vectors) are left unchanged.

The transformation can be combined with momentum, weight decay, Nesterov.
Three modes are supported (Appendix C, D):
    - 'svd'      : exact truncated SVD (default, most accurate)
    - 'rsvd'     : Randomized SVD with truncation rank `rank` (Algorithm 3)
    - 'col_norm' : column normalization (simplified EGD, SVD-free)

The `enabled` attribute can be toggled externally (e.g. by a Trainer that
detects grokking via val acc threshold) to fall back to plain SGD updates
(Algorithm 1, line 8: "Stop EGD updates after threshold is reached").
"""

from __future__ import annotations
import torch as t
from torch.optim.optimizer import Optimizer


def randomized_svd(G: t.Tensor, rank: int, n_iter: int = 2) -> tuple[t.Tensor, t.Tensor]:
    """Randomized SVD returning (U, S) only — matches Algorithm 3 of the paper.

    Batched: G can have shape (..., m, n); leading dims are treated as batch.
    No oversampling (paper says oversampling is left as future work).
    """
    *batch, m, n = G.shape
    rank = min(rank, m, n)
    Q = t.randn(*batch, n, rank, device=G.device, dtype=G.dtype)
    GT = G.transpose(-1, -2)
    for _ in range(n_iter):
        Q = t.linalg.qr(G @ Q, mode='reduced')[0]   # (..., m, r)
        Q = t.linalg.qr(GT @ Q, mode='reduced')[0]  # (..., n, r)
    B = G @ Q                                       # (..., m, r)
    U_hat, S, _ = t.linalg.svd(B, full_matrices=False)
    return U_hat, S


def egd_precondition(
    G: t.Tensor,
    mode: str = 'svd',
    rank: int | None = None,
    n_iter: int = 2,
    epsilon: float = 1e-6,
) -> t.Tensor:
    """Apply the EGD transformation to a (possibly batched) gradient matrix.

    G must have shape (..., m, n). Leading dims are treated as a batch — each
    (m, n) slice gets its own independent EGD preconditioning. This matches
    Muon's batched Newton-Schulz behavior for 3-D attention weights
    (num_heads, d_head, d_model), so head h is preconditioned with its own
    SVD of G[h] rather than mixed with the other heads.

    Returns G_tilde = (G G^T)^{-1/2} G = U V^T  (Algorithm 2 of the paper).
    For mode='col_norm', each (m, n) slice is divided column-wise by its
    column L2 norms (simplified, SVD-free variant — Section 1 / Figure 1).
    """
    if mode == 'col_norm':
        col_scales = t.linalg.norm(G, dim=-2).clamp(min=epsilon)  # (..., n)
        return G / col_scales.unsqueeze(-2)

    if mode == 'rsvd':
        assert rank is not None, "EGD rsvd mode requires a `rank` argument."
        U, S = randomized_svd(G, rank=rank, n_iter=n_iter)
    elif mode == 'svd':
        # NaN/Inf in G means training has diverged. cuSOLVER would otherwise
        # raise an opaque "failed to converge" error — raise a clean one here.
        if not t.isfinite(G).all():
            raise RuntimeError(
                "EGD: gradient is non-finite (NaN/Inf) — training diverged. "
                "Typically means lr is too high relative to wd."
            )
        try:
            U, S, _ = t.linalg.svd(G, full_matrices=False)  # truncated form
        except t._C._LinAlgError:
            # cuSOLVER occasionally fails on finite but ill-conditioned matrices
            # (known PyTorch issue). CPU LAPACK is significantly more robust.
            U_cpu, S_cpu, _ = t.linalg.svd(G.detach().cpu(), full_matrices=False)
            U, S = U_cpu.to(G.device), S_cpu.to(G.device)
    else:
        raise ValueError(f"Unknown EGD mode: {mode!r}. Use 'svd', 'rsvd', or 'col_norm'.")

    S_inv = 1.0 / S.clamp(min=epsilon)
    # P = U diag(S^{-1}) U^T  (Algorithm 2 line 8), G_tilde = P @ G  (line 9).
    # Broadcasting: U (..., m, k), S_inv.unsqueeze(-2) → (..., 1, k).
    return (U * S_inv.unsqueeze(-2)) @ (U.transpose(-1, -2) @ G)


class EGD(Optimizer):
    """Egalitarian Gradient Descent.

    Arguments
    ---------
    params           : iterable of params or param-groups.
    lr               : learning rate.
    momentum         : SGD momentum coefficient (default 0).
    weight_decay     : L2 / decoupled WD coefficient (default 0).
    nesterov         : use Nesterov-style momentum (default False).
    decoupled_wd     : if True, AdamW-style decoupled WD `p *= (1 - lr*wd)`;
                       if False, coupled WD via `g += wd * p` (default False,
                       matches paper experiments).
    mode             : 'svd' | 'rsvd' | 'col_norm' (default 'svd').
    rank             : truncation rank for RSVD (required for mode='rsvd').
    n_iter           : power iterations for RSVD (default 2).
    epsilon          : floor for singular values (Remark 2 / rank-deficient
                       case in the paper; default 1e-6).
    min_ndim         : only apply EGD to params with `p.ndim >= min_ndim`;
                       lower-ndim grads pass through unchanged (default 2).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        decoupled_wd: bool = False,
        mode: str = 'svd',
        rank: int | None = None,
        n_iter: int = 2,
        epsilon: float = 1e-6,
        min_ndim: int = 2,
    ):
        if mode == 'rsvd' and rank is None:
            raise ValueError("EGD: mode='rsvd' requires `rank=<int>`.")
        if nesterov and momentum <= 0:
            raise ValueError("EGD: Nesterov requires momentum > 0.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            decoupled_wd=decoupled_wd,
            mode=mode,
            rank=rank,
            n_iter=n_iter,
            epsilon=epsilon,
            min_ndim=min_ndim,
        )
        super().__init__(params, defaults)
        # Toggle that can be flipped to False by a Trainer once grokking is detected
        # (Algorithm 1, line 8 of the paper).
        self.enabled: bool = True

    @t.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr           = group['lr']
            momentum     = group['momentum']
            wd           = group['weight_decay']
            nesterov     = group['nesterov']
            decoupled_wd = group['decoupled_wd']
            mode         = group['mode']
            rank         = group['rank']
            n_iter       = group['n_iter']
            epsilon      = group['epsilon']
            min_ndim     = group['min_ndim']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                if self.enabled and p.ndim >= min_ndim:
                    # `egd_precondition` natively supports batched input of
                    # shape (..., m, n) — leading dims (e.g. num_heads for
                    # W_K/W_Q/W_V of shape (heads, d_head, d_model)) are
                    # treated as batch, so each head gets its own SVD. This
                    # matches Muon's per-head Newton-Schulz orthogonalization.
                    g = egd_precondition(
                        g, mode=mode, rank=rank, n_iter=n_iter, epsilon=epsilon,
                    )

                # Coupled WD (paper-style: WD added to loss => grad += wd * p)
                if wd != 0 and not decoupled_wd:
                    g = g.add(p, alpha=wd)

                # Momentum (SGD-style)
                if momentum != 0:
                    state = self.state[p]
                    buf = state.get('momentum_buffer')
                    if buf is None:
                        buf = t.clone(g).detach()
                        state['momentum_buffer'] = buf
                    else:
                        buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if nesterov else buf

                # Decoupled WD (AdamW-style: shrink p directly)
                if wd != 0 and decoupled_wd:
                    p.mul_(1 - lr * wd)

                p.add_(g, alpha=-lr)

        return loss
