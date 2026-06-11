from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import torch as t
import torch.nn as nn
from optimizer.muon import SingleDeviceMuon as Muon  # single-GPU variant of Keller Jordan's Muon
from optimizer.soap import SOAP
from optimizer.egd  import EGD                        # PyTorch port of Pasand & Dohmatob (ICLR 2026)


ParamFilter = Callable[[str, t.Tensor], bool] 



@dataclass(frozen=True)
class OptimizerSpec:
    """
    description of one optimizer applied to a subset of model parameters.

    Fields
    ------
    name : str
        Identifier of the algorithm. Currently supported: 'adamw', 'muon', 'sgd', 'soap', 'egd'.
        Adding a new optimizer = adding one entry in `_OPTIMIZER_REGISTRY`.

    lr : float
        Learning rate.

    weight_decay : float, default 0.0
        Decoupled weight decay coefficient (AdamW-style for both AdamW and Muon).

    #! param_filter : Optional[Callable[(name, tensor), bool]], default None
        predicate selecting which parameters this spec governs.
        If None, this spec acts as a *catch-all*: it captures every parameter
        not already claimed by an earlier spec.
        If multiple specs match a parameter, the FIRST spec in the list wins.

    extra : dict
        Algorithm-specific hyperparameters (e.g. {'betas': (0.9, 0.98)} for AdamW,
        {'momentum': 0.95, 'nesterov': True, 'ns_steps': 5} for Muon).

    """
    name: str
    lr: float
    weight_decay: float = 0.0
    param_filter: Optional[ParamFilter] = None
    extra: dict = field(default_factory=dict)  #! ensure default is a new dict for each instance

    def describe(self) -> str:
        """Short human-readable summary for logs."""
        kv = f"lr={self.lr}, wd={self.weight_decay}"
        if self.extra:
            kv += ", " + ", ".join(f"{k}={v}" for k, v in self.extra.items())
        scope = "ALL" if self.param_filter is None else "filtered"
        return f"{self.name}({scope}, {kv})"




# Contains all the optimizers we will use 
#! ==> to be completed with soap etc... !!
_OPTIMIZER_REGISTRY: dict[str, Callable] = {}
#! On pourrait mettre une fonction pour ajouter optimiser depuis API mais autant tout faire direct ici... 

def _register_default_optimizers():
    """Populate the registry with the optimizers supported out of the box."""
    import torch.optim as optim

    def _build_adamw(params, lr, weight_decay, **extra):
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **extra)

    def _build_muon(params, lr, weight_decay, **extra):
        return Muon(params, lr=lr, weight_decay=weight_decay, **extra)  

    def _build_sgd(params, lr, weight_decay, **extra):
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, **extra)

    def _build_soap(params, lr, weight_decay, **extra):
        return SOAP(params, lr=lr, weight_decay=weight_decay, **extra)

    def _build_egd(params, lr, weight_decay, **extra):
        return EGD(params, lr=lr, weight_decay=weight_decay, **extra)

    #! add new optimizers here !!

    _OPTIMIZER_REGISTRY['adamw'] = _build_adamw
    _OPTIMIZER_REGISTRY['muon']  = _build_muon
    _OPTIMIZER_REGISTRY['sgd']   = _build_sgd
    _OPTIMIZER_REGISTRY['soap']  = _build_soap
    _OPTIMIZER_REGISTRY['egd']   = _build_egd


_register_default_optimizers()

def build_optimizers(
    model: nn.Module,
    specs: list[OptimizerSpec],
    verbose: bool = False,
) -> list[t.optim.Optimizer]:
    """Build one optimizer per spec by partitioning model parameters.

    #! Catch-all specs (param_filter=None) are automatically moved to the end. Each parameter is assigned to the first spec whose filter returns True.

    Example
    -------
        specs = [
            OptimizerSpec("muon",  lr=0.01,  param_filter=lambda n, _: "attn" in n),
            OptimizerSpec("adamw", lr=0.001, param_filter=None),  # catch-all
        ]
        optimizers = build_optimizers(model, specs, verbose=True)
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()

    Raises
    ------
    ValueError
        If specs is empty, if a parameter is not claimed by any spec,
        or if an optimizer name is not found in the registry.
    """
    if not specs:
        raise ValueError("build_optimizers received an empty specs list.")

    # Automatically move catch-all specs to the end, preserving original order
    specs = sorted(specs, key=lambda s: s.param_filter is None)

    if verbose:
        print("Spec order after sorting:")
        for i, spec in enumerate(specs):
            print(f"  {i} : {spec.describe()}")

    groups:      list[list[t.Tensor]] = [[] for _ in specs]
    group_names: list[list[str]]      = [[] for _ in specs]

    for name, p in model.named_parameters():
        claimed_idx = None
        for i, spec in enumerate(specs):
            if spec.param_filter is None or spec.param_filter(name, p):
                claimed_idx = i
                break
        if claimed_idx is None:
            raise ValueError(
                f"Parameter '{name}' (shape={tuple(p.shape)}) is not claimed by "
                f"any OptimizerSpec. Add a catch-all spec (param_filter=None) "
                f"or widen an existing filter."
            )
        groups[claimed_idx].append(p)
        group_names[claimed_idx].append(name)

    optimizers: list[t.optim.Optimizer] = []
    for spec, params, names in zip(specs, groups, group_names):
        if not params:
            if verbose:
                print(f"  ! {spec.describe()} captured 0 params -> skipped")
            continue
        if spec.name not in _OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unknown optimizer name: '{spec.name}'. "
                f"Known: {sorted(_OPTIMIZER_REGISTRY.keys())}"
            )
        opt = _OPTIMIZER_REGISTRY[spec.name](
            params, lr=spec.lr, weight_decay=spec.weight_decay, **spec.extra,
        )
        optimizers.append(opt)
        if verbose:
            n_params = sum(p.numel() for p in params)
            print(f"  {spec.describe()}: {len(params)} tensors, {n_params:,} params")
            for nm in names:
                print(f"      |__ {nm}")

    return optimizers



_MODEL_BACKEND: str = 'vanilla' #! tl pour le model de jean avec les hooks


def set_model_backend(backend: str) -> None:
    """Switch the model implementation used by the pipeline.
    """
    global _MODEL_BACKEND # to ensure we modify the global variable 
    if backend not in {'vanilla', 'tl'}:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'vanilla' or 'tl'.")
    _MODEL_BACKEND = backend


def build_model(config) -> nn.Module:
    """Instantiate the transformer used by the pipeline.
    """
    if _MODEL_BACKEND == 'vanilla':
        from model import Transformer  # local import keeps top-level cheap
        return Transformer(config, use_cache=False)

    if _MODEL_BACKEND == 'tl':
        #! model avec les hookers, mais en vrai faudra tej distinction entre les deux c'est juste au cas ou ca demande des traitements très différent
        raise NotImplementedError(
            "Model avec Hook encore sur la planche !!! Avanti le J"
        )
    raise RuntimeError(f"Unreachable: backend={_MODEL_BACKEND!r}")


def cross_entropy_high_precision(logits: t.Tensor, labels: t.Tensor) -> t.Tensor:
    """Cross-entropy loss in float64 (float32 on MPS) for stability at low loss.
    """
    target_dtype = t.float32 if logits.device.type == 'mps' else t.float64 #! est ce que on va devoir tout run sur cuda !! :(
    logprobs = t.nn.functional.log_softmax(logits.to(target_dtype), dim=-1)
    prediction_logprobs = t.gather(logprobs, index=labels[:, None], dim=-1)
    return -t.mean(prediction_logprobs)


def gen_train_test(config) -> tuple[list, list]:
    """Generate a reproducible train/test split of the modular addition task.
    """
    import random
    p = config.p
    pairs = [(i, j, p) for i in range(p) for j in range(p)]
    random.seed(config.seed)
    random.shuffle(pairs)
    div = int(config.frac_train * len(pairs))
    return pairs[:div], pairs[div:]


# =============================================================================

class Trainer:
    """Stateful trainer for grokking experiments
    """
    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Set Python / NumPy / torch RNG seeds (CPU + CUDA)."""
        import random as random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)

    @staticmethod
    def _labels_for(data: t.Tensor, p: int) -> t.Tensor:
        """Compute labels (a + b) mod p for a batch of (a, b, =) triplets."""
        return (data[:, 0] + data[:, 1]) % p

    @staticmethod
    def _empty_history(label: str, specs: list[OptimizerSpec], seed: int) -> dict:
        """Initialize the history dict with all the keys we may populate."""
        return {
            # Metadata (for joining results later)
            'label': label,
            'seed': seed,
            'specs_repr': [s.describe() for s in specs],
            # Fast eval (every `eval_every` epochs)
            'epoch': [], 'train_loss': [], 'test_loss': [],
            'train_acc': [], 'test_acc': [],
            'l2_norm': [],   # sum_i ||W_i||^2 — cheap, tracked at eval frequency
            # L2 decomposed by module group (Nanda cleanup analysis)
            'l2_embed': [], 'l2_attn': [], 'l2_mlp': [], 'l2_unembed': [],
            # Optimizer dynamics (captured on the step immediately before each eval)
            'grad_norm': [],            # ||grad|| of the last step
            'update_norm': [],          # ||theta_{t+1} - theta_t|| over all params
            'update_norm_per_opt': [],  # same, decomposed per optimizer
            # Wall-clock seconds since the previous eval (epoch=0 entry: since init)
            'eval_wallclock': [],
            # Fourier progress measures (every `fourier_every` epochs)
            'fourier_epoch': [], 'key_freqs': [],
            'restricted_loss': [], 'excluded_loss': [], 'excluded_loss_mean': [],
            'gini_W_E': [], 'gini_W_L': [],
            'restricted_loss_all': [], 'restricted_acc_all': [],
            'restricted_loss_train': [], 'restricted_acc_train': [],
            'restricted_loss_test': [], 'restricted_acc_test': [],
            'excluded_all_loss_all': [], 'excluded_all_acc_all': [],
            'excluded_all_loss_train': [], 'excluded_all_acc_train': [],
            'excluded_all_loss_test': [], 'excluded_all_acc_test': [],
            'wl_top5_concentration': [], 'we_top5_concentration': [],
            'wl_entropy': [], 'we_entropy': [],
            'wl_frequency_masses': [], 'we_frequency_masses': [],
            'cos_coefficients': [], 'key_cos_coefficients': [],
        }

    def __init__(
        self,
        config,
        specs: list[OptimizerSpec],
        seed: int = 0,
        label: str = 'run',
        *, #! force keyword arguments after this point for clarity in calls to Trainer()
        eval_every: int = 50,
        fourier_every: Optional[int] = None,   # None to disable Fourier metrics
        fixed_key_freqs: Optional[list] = None, # if set, fourier_metrics uses these freqs (else adaptive per snapshot)
        warmup_steps: int = 10,
        verbose_every: int = 5000,
        verbose_build: bool = False,
    ):
        Trainer._seed_everything(seed)
        self.config = config
        self.specs = specs
        self.seed = seed
        self.label = label
        self.eval_every = eval_every
        self.fourier_every = fourier_every
        self.fixed_key_freqs = fixed_key_freqs
        self.warmup_steps = warmup_steps
        self.verbose_every = verbose_every

        device = config.device
        self.device = device

        train_pairs, test_pairs = gen_train_test(config)
        self.train_pairs = train_pairs
        self.test_pairs  = test_pairs
        self.train_data = t.tensor(train_pairs, dtype=t.long, device=device)
        self.test_data  = t.tensor(test_pairs,  dtype=t.long, device=device)
        self.all_data   = t.tensor(
            [(i, j, config.p) for i in range(config.p) for j in range(config.p)],
            dtype=t.long, device=device,
        )
 
        # ============== FOURIER METRICS SETUP ==============
        self.fourier_ready = False
        self._fourier_metrics = None
        self.is_train = None
        self.is_test = None
        
        if fourier_every is not None:
            try:
                from fourier_metrics import fourier_metrics, get_train_test_masks
                self.is_train, self.is_test = get_train_test_masks(
                    train_pairs, test_pairs, config.p, device=device
                )
                self._fourier_metrics = fourier_metrics
                self.fourier_ready = True
            except Exception as ex:
                print(f"[Trainer '{label}'] Fourier metrics disabled: {ex}")
        # ====================================================
    
    
        self.model = build_model(config).to(device)
        if verbose_build:
            print(f"\n=== Trainer '{label}' (seed={seed}) ===")
            for s in specs:
                print(f"  spec: {s.describe()}")
        self.optimizers = build_optimizers(self.model, specs, verbose=verbose_build)

        #! reprends la meme que celle dans baseline.iynb !!
        warmup_fn = lambda step: min(step / max(1, warmup_steps), 1.0)
        self.schedulers = [
            t.optim.lr_scheduler.LambdaLR(opt, warmup_fn) for opt in self.optimizers
        ]

        # --- Mutable training state ---
        self.epoch: int = 0
        self.history: dict = Trainer._empty_history(label=label, specs=specs, seed=seed)
        self._callbacks: dict[str, list[Callable]] = { #! vide pour l'instant, à remplir avec les fonctions qu'on veut appeler à chaque étape de la boucle d'entrainement
            'on_eval': [],
            'on_fourier_snapshot': [],
            'on_epoch_end': [],
        }

        # --- Optimizer diagnostics setup ---
        # Map each parameter (by id) to which optimizer owns it, so update_norm
        # can be decomposed per-optimizer for hybrid setups (Muon + AdamW).
        self._opt_param_id_sets: list[set] = [
            {id(p) for group in opt.param_groups for p in group['params']}
            for opt in self.optimizers
        ]
        # Buffers populated in step() (only on steps just before an eval), read
        # in _take_eval_snapshot. None at epoch=0 (no step has happened yet).
        self._last_grad_norm: Optional[float] = None
        self._last_update_norm: Optional[float] = None
        self._last_update_norm_per_opt: Optional[list[float]] = None
        # Wall-clock anchor for the next eval delta.
        import time as _time
        self._last_eval_walltime: float = _time.perf_counter()

    def step(self) -> float:
        """One training step (forward + backward + optimizer step + scheduler step).

        On the step whose result will be evaluated next (i.e. (epoch+1) % eval_every == 0),
        also captures ||grad|| and ||update|| (total and per-optimizer) into
        self._last_*; these are consumed by the next _take_eval_snapshot.
        """
        self.model.train()

        # Decide whether to capture diagnostics this step (only ~1 step in eval_every).
        capture_diag = ((self.epoch + 1) % self.eval_every == 0)
        if capture_diag:
            # Clone before the update so we can compute ||theta_after - theta_before||.
            params_before = [p.detach().clone() for p in self.model.parameters()]

        logits = self.model(self.train_data)[:, -1, :self.config.p]
        labels = Trainer._labels_for(self.train_data, self.config.p)
        loss = cross_entropy_high_precision(logits, labels)
        loss.backward()

        if capture_diag:
            # ||grad|| over all params (post-backward, pre-step).
            gn_sq_t = t.zeros((), device=self.device)
            for p in self.model.parameters():
                if p.grad is not None:
                    gn_sq_t = gn_sq_t + p.grad.detach().pow(2).sum()
            self._last_grad_norm = gn_sq_t.sqrt().item()

        for opt in self.optimizers:
            opt.step()
        for sch in self.schedulers:
            sch.step()

        if capture_diag:
            # ||update|| total + decomposed per optimizer.
            n_opt = len(self.optimizers)
            un_per_opt_t = [t.zeros((), device=self.device) for _ in range(n_opt)]
            total_sq_t   = t.zeros((), device=self.device)
            for p, before in zip(self.model.parameters(), params_before):
                d_sq = (p.detach() - before).pow(2).sum()
                total_sq_t = total_sq_t + d_sq
                pid = id(p)
                for i, ids in enumerate(self._opt_param_id_sets):
                    if pid in ids:
                        un_per_opt_t[i] = un_per_opt_t[i] + d_sq
                        break
            self._last_update_norm = total_sq_t.sqrt().item()
            self._last_update_norm_per_opt = [u.sqrt().item() for u in un_per_opt_t]

        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        return loss.item()

    @t.no_grad()
    def evaluate(self, data: Optional[t.Tensor] = None) -> tuple[float, float]:
        """Evaluate model on `data` (defaults to test set). Returns (loss, acc)."""
        if data is None:
            data = self.test_data
        p = self.config.p
        labels = Trainer._labels_for(data, p)
        logits = self.model(data)[:, -1, :p]
        loss = cross_entropy_high_precision(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss.item(), acc.item()

    def fit(self, num_epochs: Optional[int] = None) -> dict: #! One single Training loop with eval and fourier (to do) callbacks
        """Run the full training loop.

        Resumes from `self.epoch` if called more than once, so you can:
            trainer.fit(num_epochs=5000)    # phase 1
            # ... inspect trainer.model ...
            trainer.fit(num_epochs=40000)   # continue to 40000

        Returns the (still-mutable) `history` dict.
        """
        n_epochs = num_epochs if num_epochs is not None else self.config.num_epochs
        start_epoch = self.epoch

        # ── Adaptive logging state ─────────────────────────────────────────────
        # If config.adaptive_logging is True, eval_every and fourier_every are multiplied
        # by config.adaptive_logging_factor once test_acc crosses config.adaptive_logging_thresh.
        adaptive       = getattr(self.config, 'adaptive_logging', False)
        adaptive_thresh = getattr(self.config, 'adaptive_logging_thresh', 0.99)
        adaptive_factor = getattr(self.config, 'adaptive_logging_factor', 10)
        self._adaptive_switched = False

        for self.epoch in range(start_epoch, n_epochs + 1):
            # Eval snapshot
            if self.epoch % self.eval_every == 0:
                self._take_eval_snapshot() #! definition below

                # ── Adaptive switch : check after each eval snapshot ──────────
                if (adaptive and not self._adaptive_switched
                        and self.history.get('test_acc')
                        and self.history['test_acc'][-1] >= adaptive_thresh):
                    self.eval_every = int(self.eval_every * adaptive_factor)
                    if self.fourier_every is not None:
                        self.fourier_every = int(self.fourier_every * adaptive_factor)
                    self._adaptive_switched = True
                    print(f'  [{self.label}] adaptive_logging : grok at epoch {self.epoch}, '
                          f'switching to eval_every={self.eval_every}, fourier_every={self.fourier_every}')

            # Fourier snapshot
            if self.fourier_ready and self.epoch % self.fourier_every == 0:
                self._take_fourier_snapshot() #! definition below

            if self.epoch == n_epochs:
                break

            self.step()
            self._dispatch('on_epoch_end')

        return self.history

    def register_callback(self, event: str, fn: Callable): #! fait parti de l'API publique donc on pourra ajouer ce que on veut ici
        """Register a callback for one of the lifecycle events.
        Valid events: 'on_eval', 'on_fourier_snapshot', 'on_epoch_end'.
        """
        if event not in self._callbacks:
            raise ValueError(
                f"Unknown event '{event}'. Valid: {list(self._callbacks)}"
            )
        self._callbacks[event].append(fn)

    def _dispatch(self, event: str):
        """Call all callbacks registered to `event`. Each receives `self`."""
        for cb in self._callbacks[event]:
            cb(self)

    def _take_eval_snapshot(self):
        import time as _time
        self.model.eval()
        tr_l, tr_a = self.evaluate(self.train_data)
        te_l, te_a = self.evaluate(self.test_data)

        # L2 total + decomposition by module group.
        l2_embed = l2_attn = l2_mlp = l2_unembed = 0.0
        for name, p in self.model.named_parameters():
            sq = p.detach().pow(2).sum().item()
            if 'unembed' in name:
                l2_unembed += sq
            elif 'embed' in name:                 # covers W_E and W_pos
                l2_embed += sq
            elif 'attn' in name:
                l2_attn += sq
            elif 'mlp' in name:
                l2_mlp += sq
        l2 = l2_embed + l2_attn + l2_mlp + l2_unembed

        self.history['epoch'].append(self.epoch)
        self.history['train_loss'].append(tr_l)
        self.history['test_loss'].append(te_l)
        self.history['train_acc'].append(tr_a)
        self.history['test_acc'].append(te_a)
        self.history['l2_norm'].append(l2)
        self.history['l2_embed'].append(l2_embed)
        self.history['l2_attn'].append(l2_attn)
        self.history['l2_mlp'].append(l2_mlp)
        self.history['l2_unembed'].append(l2_unembed)

        # Optimizer diagnostics captured in step() (None at epoch=0).
        self.history['grad_norm'].append(self._last_grad_norm)
        self.history['update_norm'].append(self._last_update_norm)
        self.history['update_norm_per_opt'].append(self._last_update_norm_per_opt)

        # Wall-clock seconds elapsed since the previous eval.
        now = _time.perf_counter()
        self.history['eval_wallclock'].append(now - self._last_eval_walltime)
        self._last_eval_walltime = now

        if self.verbose_every and self.epoch % self.verbose_every == 0:
            print(f"  [{self.label} seed={self.seed}] epoch {self.epoch:5d} "
                  f"| train acc {tr_a:.3f} | test acc {te_a:.3f}")
        self._dispatch('on_eval')

    def _take_fourier_snapshot(self):
        self.model.eval()
        try:
            fm_kwargs = {}
            if self.fixed_key_freqs is not None:
                fm_kwargs['key_freqs'] = self.fixed_key_freqs
            fm = self._fourier_metrics(
                self.model, self.config, self.all_data,
                self.is_train, self.is_test,
                **fm_kwargs,
            )
            self.history['fourier_epoch'].append(self.epoch)
            # NOTE: 'l2_norm' is no longer logged here — it's tracked at every
            # eval_every step in _take_eval_snapshot (cheaper + finer granularity).
            for k, v in fm.items():
                self.history.setdefault(k, []).append(v)
            self._dispatch('on_fourier_snapshot')
        except Exception as ex:
            if self.epoch == 0:
                print(f"  [{self.label}] Fourier metrics raised at epoch 0: {ex}")


    def save_run(self, folder: str, include_optimizers: bool = True) -> None:
        """Save the run as a self-contained folder with human-readable metadata.
        """
        import os
        import json
        import dataclasses
        from datetime import datetime

        os.makedirs(folder, exist_ok=True)

        # --- meta.json (config + specs + metadata) ---
        cfg_dict = dataclasses.asdict(self.config)
        cfg_dict.pop('device', None)  # device is re-detected at load time

        meta = {
            'label':    self.label,
            'seed':     self.seed,
            'epoch':    self.epoch,
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'config':   cfg_dict,
            'specs': [
                {
                    'name':              s.name,
                    'lr':                s.lr,
                    'weight_decay':      s.weight_decay,
                    'extra':             s.extra,
                    'has_param_filter':  s.param_filter is not None,
                }
                for s in self.specs
            ],
        }
        with open(os.path.join(folder, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        with open(os.path.join(folder, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        t.save(self.model.state_dict(), os.path.join(folder, 'model.pt'))
        if include_optimizers:
            t.save({
                'optimizer_states': [opt.state_dict() for opt in self.optimizers],
                'scheduler_states': [sch.state_dict() for sch in self.schedulers],
            }, os.path.join(folder, 'optim.pt'))

    @classmethod
    def from_run(
        cls,
        folder: str,
        specs: Optional[list[OptimizerSpec]] = None,
        fourier_every: Optional[int] = None,
    ) -> 'Trainer':
        """Reconstruct a Trainer from a folder produced by `save_run`.
        """
        import os
        import json

        # --- Read meta + history ---
        with open(os.path.join(folder, 'meta.json')) as f:
            meta = json.load(f)
        with open(os.path.join(folder, 'history.json')) as f:
            history = json.load(f)

        # --- Reconstruct Config (device is re-detected) ---
        from model import Config
        config = Config(**meta['config'])

        # --- Reconstruct specs if not provided ---
        if specs is None:
            spec_dicts = meta['specs']
            if any(s['has_param_filter'] for s in spec_dicts):
                raise RuntimeError(
                    f"Run at '{folder}' used param_filters (e.g. Muon hybrid). "
                    "Pass `specs=` explicitly when reloading — lambdas can't be "
                    "auto-serialized."
                )
            specs = [
                OptimizerSpec(
                    name=s['name'],
                    lr=s['lr'],
                    weight_decay=s['weight_decay'],
                    extra=dict(s['extra']),
                )
                for s in spec_dicts
            ]

        # --- Build trainer with same setup ---
        trainer = cls(
            config, specs,
            seed=int(meta['seed']),
            label=meta['label'],
            fourier_every=fourier_every,
            verbose_build=False,
        )

        # --- Restore model + epoch + history ---
        trainer.model.load_state_dict(
            t.load(os.path.join(folder, 'model.pt'), map_location=config.device)
        )
        trainer.epoch = int(meta['epoch'])
        trainer.history = history

        # --- Restore optimizers + schedulers if present ---
        optim_path = os.path.join(folder, 'optim.pt')
        if os.path.exists(optim_path):
            opt_ckpt = t.load(optim_path, map_location=config.device)
            if len(opt_ckpt['optimizer_states']) != len(trainer.optimizers):
                raise RuntimeError(
                    f"Mismatch: checkpoint has {len(opt_ckpt['optimizer_states'])} "
                    f"optimizers, current Trainer has {len(trainer.optimizers)}. "
                    "Specs must match the saved run."
                )
            for opt, state in zip(trainer.optimizers, opt_ckpt['optimizer_states']):
                opt.load_state_dict(state)
            for sch, state in zip(trainer.schedulers, opt_ckpt['scheduler_states']):
                sch.load_state_dict(state)

        return trainer


# =============================================================================
# Multi-seed runner
# =============================================================================

def run_multi_seed(
    config,
    specs: list[OptimizerSpec],
    seeds: list[int],
    label_prefix: str = 'run',
    save_root: Optional[str] = None,
    **trainer_kwargs,
) -> dict[int, Trainer]:
    """Run the same config across multiple seeds.

    Each seed gets a fresh Trainer (independent model init + data shuffle),
    trained for the full `config.num_epochs`. Returns {seed: trainer}.
    """
    results: dict[int, Trainer] = {}
    for i, seed in enumerate(seeds):
        print(f"\n=== {label_prefix} seed {seed} ({i+1}/{len(seeds)}) ===")
        trainer = Trainer(
            config, specs,
            seed=seed,
            label=f"{label_prefix}_seed{seed}",
            **trainer_kwargs,
        )
        diverged = False
        try:
            trainer.fit()
        except RuntimeError as e:
            msg = str(e)
            if 'EGD' in msg or 'non-finite' in msg.lower() or 'diverged' in msg.lower():
                print(f"  ⚠ seed {seed} DIVERGED at epoch {trainer.epoch}: {msg}")
                diverged = True
            else:
                raise
        if save_root is not None:
            if diverged and isinstance(trainer.history, dict):
                trainer.history['diverged'] = True
            trainer.save_run(f"{save_root}/seed{seed}")
            print(f"  saved to {save_root}/seed{seed}/{' [DIVERGED]' if diverged else ''}")
        results[seed] = trainer
    return results






# =============================================================================
# Grid search over hyperparameters
# =============================================================================

def first_epoch_above(history: dict, key: str, threshold: float) -> Optional[int]:
    """Return the first epoch where history[key] crosses `threshold` (None if never)."""
    for e, v in zip(history['epoch'], history[key]):
        if v > threshold:
            return e
    return None


def _combo_label(combo: tuple, keys: list[str]) -> str:
    """Convert a (param_value, ...) tuple to a filesystem-safe folder name."""
    parts = []
    for k, v in zip(keys, combo):
        if isinstance(v, (tuple, list)):
            v_str = '-'.join(f"{x:g}" if isinstance(x, float) else str(x) for x in v)
        elif isinstance(v, float):
            v_str = f"{v:g}"
        else:
            v_str = str(v)
        v_str = (v_str
                 .replace('/', '_')
                 .replace(' ', '')
                 .replace('(', '')
                 .replace(')', '')
                 .replace(',', ''))
        parts.append(f"{k}{v_str}")
    return '__'.join(parts)



def grid_search(
    config,
    spec_builder: Callable[..., list[OptimizerSpec]],
    param_grid: dict[str, list],
    seeds: list[int],
    save_root: str,
    num_epochs: Optional[int] = None,
    **trainer_kwargs,
) -> dict[tuple, dict[int, Trainer]]:
    """Grid search over `param_grid` × seeds. Resume-tolerant.

    Parameters
    ----------
    config : Config
        Base configuration. If `num_epochs` is given, the config is replaced
        with `num_epochs=num_epochs` (typical use: 25k for phase 1 search).
    spec_builder : Callable
        Function that takes named kwargs (matching `param_grid` keys) and
        returns a list of OptimizerSpec. This is the modular extension point:
        you can build any spec configuration from any set of hyperparameters.

        Example for AdamW:
            def make_adamw(lr, weight_decay):
                return [OptimizerSpec('adamw', lr=lr, weight_decay=weight_decay,
                                       extra={'betas': (0.9, 0.98)})]

        Example for Muon hybrid:
            def make_muon_hybrid(lr_muon, wd_muon):
                return [
                    OptimizerSpec('muon', lr=lr_muon, weight_decay=wd_muon,
                                  param_filter=lambda n, p: p.ndim >= 2
                                                            and 'embed' not in n,
                                  extra={'momentum': 0.95}),
                    OptimizerSpec('adamw', lr=1e-3, weight_decay=1.0,
                                  extra={'betas': (0.9, 0.98)}),
                ]

    param_grid : dict[str, list]
        Maps parameter name to list of values. Cartesian product is iterated.
        Example: {'lr': [1e-3, 5e-3], 'weight_decay': [0.3, 1.0]}

    seeds : list[int]
        Seeds to run per combination (typically 2-3 in phase 1, 5+ in phase 2).

    save_root : str
        Parent folder. Each combination is saved at
            <save_root>/<combo_label>/seed{N}/
        which makes resume trivial.

    num_epochs : Optional[int]
        Override config.num_epochs (e.g. 25_000 for phase 1).

    **trainer_kwargs
        Forwarded to Trainer (eval_every, fourier_every, warmup_steps,
        verbose_every, verbose_build).

    Returns
    -------
    dict[tuple, dict[int, Trainer]]
        {combo_tuple: {seed: trainer}}. Use `aggregate_grid` to analyse.

    Notes
    -----
    Resume: if a seed folder already exists, it is reloaded via Trainer.from_run
    (with the same `spec_builder` to handle param_filter lambdas). Only missing
    seeds are actually trained.
    """
    import os
    import itertools
    import dataclasses

    if num_epochs is not None:
        config = dataclasses.replace(config, num_epochs=num_epochs)

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    all_combos = list(itertools.product(*value_lists))

    print(f"\n=== grid_search: {len(all_combos)} combinations × {len(seeds)} seeds "
          f"= {len(all_combos) * len(seeds)} runs ===")
    print(f"   params : {keys}")
    print(f"   save   : {save_root}/")
    print()

    results: dict[tuple, dict[int, Trainer]] = {}

    for combo_idx, combo in enumerate(all_combos):
        params = dict(zip(keys, combo))
        label = _combo_label(combo, keys)
        combo_folder = os.path.join(save_root, label)

        print(f"--- [{combo_idx+1}/{len(all_combos)}] {label} ---")

        # Resume: reload existing seeds from disk
        existing: dict[int, Trainer] = {}
        if os.path.exists(combo_folder):
            specs_for_reload = spec_builder(**params)
            for d in sorted(os.listdir(combo_folder)):
                if d.startswith('seed'):
                    seed = int(d.replace('seed', ''))
                    if seed in seeds:
                        try:
                            existing[seed] = Trainer.from_run(
                                os.path.join(combo_folder, d),
                                specs=specs_for_reload,
                                fourier_every=trainer_kwargs.get('fourier_every'),
                            )
                        except Exception as ex:
                            print(f"   ⚠ could not reload seed{seed}: {ex}")

        missing = [s for s in seeds if s not in existing]
        if existing:
            print(f"   already done : {sorted(existing)}")
        if missing:
            print(f"   will run     : {missing}")

        if missing:
            specs = spec_builder(**params)
            new_runs = run_multi_seed(
                config, specs, seeds=missing,
                label_prefix=label,
                save_root=combo_folder,
                **trainer_kwargs,
            )
            existing.update(new_runs)

        results[combo] = existing

    print(f"\n=== grid_search complete: {len(results)} combinations done ===")
    return results


def aggregate_grid(
    grid_results: dict[tuple, dict[int, 'Trainer']],
    param_keys: list[str],
    *,
    acc_thresh: float = 0.99,
    robustness_thresh: float = 0.5,
) -> list[dict]:
    """Reduce a grid_search result to a sorted table of {params + stats}.
    """
    import numpy as np

    rows = []
    for combo, seed_results in grid_results.items():
        param_dict = dict(zip(param_keys, combo))
        epochs_mem      = []   # epoch where train_acc crosses acc_thresh
        epochs_grok     = []   # epoch where test_acc crosses acc_thresh
        gaps            = []   # epoch_grok - epoch_memorize (only when both exist)
        l2_finals       = []   # ||W||^2 at the end of training
        final_test_accs = []
        for trainer in seed_results.values():
            h = trainer.history if hasattr(trainer, 'history') else trainer
            em = first_epoch_above(h, 'train_acc', acc_thresh)
            eg = first_epoch_above(h, 'test_acc',  acc_thresh)
            if em is not None:
                epochs_mem.append(em)
            if eg is not None:
                epochs_grok.append(eg)
            if em is not None and eg is not None:
                gaps.append(eg - em)
            final_test_accs.append(h['test_acc'][-1])
            l2_hist = h.get('l2_norm', [])
            if l2_hist:
                l2_finals.append(l2_hist[-1])

        n_seeds = len(seed_results)
        n_grok = len(epochs_grok)
        ratio = n_grok / n_seeds if n_seeds else 0.0

        row = {
            **param_dict,
            'n_grok':              n_grok,
            'n_total':             n_seeds,
            'did_grok_ratio':      ratio,
            'did_grok_str':        f"{n_grok}/{n_seeds}",
            'epoch_grok_median':   int(np.median(epochs_grok)) if epochs_grok else None,
            'epoch_grok_min':      min(epochs_grok)             if epochs_grok else None,
            'epoch_grok_max':      max(epochs_grok)             if epochs_grok else None,
            'epoch_mem_median':    int(np.median(epochs_mem))   if epochs_mem else None,
            'grok_gap_median':     int(np.median(gaps))         if gaps else None,
            'l2_final_median':     round(float(np.median(l2_finals)), 1) if l2_finals else None,
            'final_test_acc_med':  round(float(np.median(final_test_accs)), 4),
        }
        rows.append(row)

    valid   = [r for r in rows if r['did_grok_ratio'] >= robustness_thresh]
    invalid = [r for r in rows if r['did_grok_ratio'] <  robustness_thresh]

    valid.sort(key=lambda r: (r['epoch_grok_median'] if r['epoch_grok_median'] is not None else 1e18))

    return valid + invalid


def print_grid_table(rows: list[dict], param_keys: list[str], top_n: Optional[int] = None) -> None:
    """Pretty-print the output of aggregate_grid.

    Shows: params + did_grok + memorization/grok epochs + gap + L2 final + final_acc.
    """
    if not rows:
        print("(no rows)")
        return
    if top_n is not None:
        rows = rows[:top_n]

    display_keys = list(param_keys) + [
        'did_grok_str',
        'epoch_mem_median', 'epoch_grok_median', 'grok_gap_median',
        'epoch_grok_min', 'epoch_grok_max',
        'l2_final_median', 'final_test_acc_med',
    ]
    headers = {
        **{k: k for k in param_keys},
        'did_grok_str':       'did_grok',
        'epoch_mem_median':   'mem_med',
        'epoch_grok_median':  'eg_med',
        'grok_gap_median':    'gap_med',
        'epoch_grok_min':     'eg_min',
        'epoch_grok_max':     'eg_max',
        'l2_final_median':    'l2_final',
        'final_test_acc_med': 'final_acc',
    }
    widths = {
        k: max(len(headers[k]), max(len(str(r.get(k, ''))) for r in rows))
        for k in display_keys
    }
    sep = ' | '
    print(sep.join(headers[k].ljust(widths[k]) for k in display_keys))
    print(sep.join('-' * widths[k] for k in display_keys))
    for r in rows:
        print(sep.join(str(r.get(k, '')).ljust(widths[k]) for k in display_keys))
