from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import torch as t
import torch.nn as nn
from old.muon import Muon  #? Maybe import the Muon 


ParamFilter = Callable[[str, t.Tensor], bool] # type alias: a filter takes (param_name, tensor) and returns whether this



@dataclass(frozen=True)
class OptimizerSpec:
    """
    description of one optimizer applied to a subset of model parameters.

    Fields
    ------
    name : str
        Identifier of the algorithm. Currently supported: 'adamw', 'muon', 'sgd'.
        Adding a new optimizer = adding one branch in `build_optimizers`.

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


def _register_default_optimizers():
    """Populate the registry with the optimizers supported out of the box."""
    import torch.optim as optim

    def _build_adamw(params, lr, weight_decay, **extra):
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **extra)

    def _build_muon(params, lr, weight_decay, **extra):
        return Muon(params, lr=lr, weight_decay=weight_decay, **extra)   #! doit pouvoir etre appelé de la sorte, avec **extra !!

    def _build_sgd(params, lr, weight_decay, **extra):
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, **extra)
    
    #! add new optimizers here !!

    _OPTIMIZER_REGISTRY['adamw'] = _build_adamw
    _OPTIMIZER_REGISTRY['muon']  = _build_muon
    _OPTIMIZER_REGISTRY['sgd']   = _build_sgd


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
            # Fourier progress measures (every `fourier_every` epochs)
            'fourier_epoch': [], 'key_freqs': [],
            'restricted_loss': [], 'excluded_loss': [], 'excluded_loss_mean': [],
            'l2_norm': [], 'gini_W_E': [], 'gini_W_L': [],
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
        
        # if fourier_every is not None:
        #     try:
        #         from helpers import fourier_metrics, get_train_test_masks
        #         is_train, is_test = get_train_test_masks(train_pairs, test_pairs, config.p)
        #         self.is_train = is_train.to(device)
        #         self.is_test  = is_test.to(device)
        #         self._fourier_metrics = fourier_metrics
        #         self.fourier_ready = True
        #     except Exception as ex:
        #         print(f"[Trainer '{label}'] Fourier metrics disabled: {ex}")
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

    def step(self) -> float:
        """One training step (forward + backward + optimizer step + scheduler step).
        """
        self.model.train()
        logits = self.model(self.train_data)[:, -1, :self.config.p]
        labels = Trainer._labels_for(self.train_data, self.config.p)
        loss = cross_entropy_high_precision(logits, labels)
        loss.backward()
        for opt in self.optimizers:
            opt.step()
        for sch in self.schedulers:
            sch.step()
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

        for self.epoch in range(start_epoch, n_epochs + 1):
            # Eval snapshot
            if self.epoch % self.eval_every == 0:
                self._take_eval_snapshot() #! definition below

            # Fourier snapshot
            if self.fourier_ready and self.epoch % self.fourier_every == 0:
                self._take_fourier_snapshot() #! definition below

            if self.epoch == n_epochs:
                break

            self.step()
            self._dispatch('on_epoch_end')

        return self.history

    def register_callback(self, event: str, fn: Callable): #! fait parti de l'API publique donc on pourra ajouer ce que on veur ici
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
        self.model.eval()
        tr_l, tr_a = self.evaluate(self.train_data)
        te_l, te_a = self.evaluate(self.test_data)
        self.history['epoch'].append(self.epoch)
        self.history['train_loss'].append(tr_l)
        self.history['test_loss'].append(te_l)
        self.history['train_acc'].append(tr_a)
        self.history['test_acc'].append(te_a)
        if self.verbose_every and self.epoch % self.verbose_every == 0:
            print(f"  [{self.label} seed={self.seed}] epoch {self.epoch:5d} "
                  f"| train acc {tr_a:.3f} | test acc {te_a:.3f}")
        self._dispatch('on_eval')

    def _take_fourier_snapshot(self):
        self.model.eval()
        try:
            fm = self._fourier_metrics(
                self.model, self.config, self.all_data,
                self.is_train, self.is_test,
            )
            self.history['fourier_epoch'].append(self.epoch)
            for k in ('key_freqs', 'restricted_loss', 'excluded_loss',
                      'excluded_loss_mean', 'l2_norm', 'gini_W_E', 'gini_W_L'):
                self.history[k].append(fm[k])
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
        trainer.fit()
        if save_root is not None:
            trainer.save_run(f"{save_root}/seed{seed}")
            print(f"  saved to {save_root}/seed{seed}/")
        results[seed] = trainer
    return results

