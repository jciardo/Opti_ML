"""One-shot script to inject a Section 1.4 diagnostic cell into the 3 phase2 notebooks.

Inserts two cells (markdown intro + code) BEFORE cell index 11 (the PHASE 2 header) in
each notebook. Existing cells are untouched.

Safe to re-run : idempotent (detects existing '## 1.4 — Diagnostic' header and skips).
"""
import json
from pathlib import Path
import uuid

ROOT = Path('/Users/mverest/Desktop/Fourier/Opti_ML/baseline')
NOTEBOOKS = ['egd_phase2.ipynb', 'muon_phase2.ipynb', 'adamw_phase2.ipynb']

MD_SOURCE = """## 1.4 — Diagnostic : comparaison des méthodes de sélection key-freqs

**Goal** : pour chaque config × seed, compare 4 méthodes pour identifier les key freqs sur W_L final :

- **topk k=5** : reference Nanda (k fixé, déjà utilisé en Phase 2)
- **cum_energy τ=0.85 / 0.90 / 0.95** : garde les freqs portant τ % de l'énergie de W_L (Parseval)
- **permutation** : test statistique vs null par row-shuffle de W_L (le plus rigoureux, n_perms=500, α=0.05, FDR)

**Conclusion à tirer** : si pour un optimizer le n_keep diffère beaucoup de 5, sa Phase 2 actuelle est sous-optimale → candidat pour re-train dans un nouveau notebook.

⚠ Cette cellule **NE MODIFIE PAS** `cfg['per_seed_freqs']` — la Phase 2 (cellule 2.1) reste inchangée. Les résultats sont stockés dans `cfg['kf_diagnostics']` (nouvelle clé)."""

CODE_SOURCE = """# Inline reload of per-seed Fourier components (no plotting, no existing-code change).
# Then apply 4 selection methods and build a comparison table.

import pandas as pd
from fourier_metrics import make_fourier_basis

_basis    = make_fourier_basis(BASE_CONFIG).cpu()
_cos_rows = _basis[1::2]
_sin_rows = _basis[2::2]

def _load_components(save_root, seeds, p):
    out = {}
    for s in seeds:
        mp = Path(save_root) / f'seed{s}' / 'model.pt'
        if not mp.exists(): continue
        state = t.load(mp, map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        def _find(suf):
            for k, v in state.items():
                if k.endswith(suf): return v
            return None
        W_E   = _find('W_E')
        W_U   = _find('W_U')
        W_out = _find('W_out')
        W_E_p = W_E[:, :p].float().cpu()
        W_L   = (W_U[:, :p].float().cpu().T) @ W_out.float().cpu()
        out[s] = {
            'we_cos':     (W_E_p @ _cos_rows.T).norm(dim=0).numpy(),
            'we_sin':     (W_E_p @ _sin_rows.T).norm(dim=0).numpy(),
            'wl_cos':     (_cos_rows @ W_L).norm(dim=1).numpy(),
            'wl_sin':     (_sin_rows @ W_L).norm(dim=1).numpy(),
            '_W_L':       W_L.numpy(),
            '_basis_cos': _cos_rows.numpy(),
            '_basis_sin': _sin_rows.numpy(),
        }
    return out

# ── Apply 4 selection methods per config × seed ────────────────────────────
ALL_ROWS = []
for name, cfg in CONFIGS.items():
    converged = cfg['converged']
    if len(converged) < 3:
        print(f'  ⚠ {name} : moins de 3 seeds convergés — skip'); continue
    comp = _load_components(cfg['phase1_dir'], converged, p=BASE_CONFIG.p)

    kf_topk = identify_key_freqs(comp, method='topk',       k=5,        verbose=False)
    kf_e85  = identify_key_freqs(comp, method='cum_energy', tau=0.85,   verbose=False)
    kf_e90  = identify_key_freqs(comp, method='cum_energy', tau=0.90,   verbose=False)
    kf_e95  = identify_key_freqs(comp, method='cum_energy', tau=0.95,   verbose=False)
    kf_perm = identify_key_freqs(comp, method='permutation',
                                  n_perms=500, alpha=0.05, correction='fdr',
                                  verbose=False)

    cfg['kf_diagnostics'] = {
        'topk_k5':         kf_topk,
        'cum_t0.85':       kf_e85,
        'cum_t0.90':       kf_e90,
        'cum_t0.95':       kf_e95,
        'permutation_fdr': kf_perm,
    }

    for s in sorted(comp.keys()):
        def _fmt(kfd):
            kf = kfd['per_seed'][s]
            return f'({len(kf):2d}) {kf}'
        ALL_ROWS.append({
            'config':         name,
            'seed':           s,
            'topk k=5':       _fmt(kf_topk),
            'cum τ=0.85':     _fmt(kf_e85),
            'cum τ=0.90':     _fmt(kf_e90),
            'cum τ=0.95':     _fmt(kf_e95),
            'permutation':    _fmt(kf_perm),
        })

df_kf = pd.DataFrame(ALL_ROWS)
print('═══ Per-seed key-freq selection — count + freq list ═══')
with pd.option_context('display.max_colwidth', None):
    display(df_kf)

# ── Summary : n_keep mean ± std per config × method ──────────────────────────
summary_rows = []
for name, cfg in CONFIGS.items():
    diag = cfg.get('kf_diagnostics')
    if diag is None: continue
    row = {'config': name}
    for method_name, kf in diag.items():
        counts = [len(v) for v in kf['per_seed'].values()]
        if counts:
            row[method_name] = f'{np.mean(counts):.1f} ± {np.std(counts):.1f}   (range {min(counts)}–{max(counts)})'
    summary_rows.append(row)

print('\\n═══ n_keep summary per config × method ═══')
display(pd.DataFrame(summary_rows))

print('\\n→ Phase 2 utilise toujours `cfg[\"per_seed_freqs\"]` (topk k=5). Inchangé.')
print('→ Diagnostics dispo dans `cfg[\"kf_diagnostics\"]` pour analyse a posteriori.')"""


def _new_cell(cell_type: str, source: str) -> dict:
    """Create a new nbformat 4.5+ cell with a fresh ID."""
    cell = {
        'cell_type': cell_type,
        'id':        uuid.uuid4().hex[:8],
        'metadata':  {},
        'source':    source.splitlines(keepends=True),
    }
    if cell_type == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    return cell


def _has_diag_cell(nb: dict) -> bool:
    """Idempotency check : has a '## 1.4 — Diagnostic' header already been inserted?"""
    for c in nb['cells']:
        src = ''.join(c['source'])
        if '## 1.4 — Diagnostic' in src:
            return True
    return False


def _find_phase2_header_idx(nb: dict) -> int:
    """Return the index of the markdown cell that starts the PHASE 2 section."""
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'markdown': continue
        src = ''.join(c['source'])
        if 'PHASE 2' in src and 'Re-train' in src:
            return i
    raise RuntimeError("PHASE 2 header cell not found")


for nb_name in NOTEBOOKS:
    path = ROOT / nb_name
    nb = json.loads(path.read_text())
    if _has_diag_cell(nb):
        print(f'  {nb_name:25s} : already has Section 1.4 — skip')
        continue
    idx = _find_phase2_header_idx(nb)
    md_cell   = _new_cell('markdown', MD_SOURCE)
    code_cell = _new_cell('code',     CODE_SOURCE)
    # Insert BEFORE the PHASE 2 header
    nb['cells'] = nb['cells'][:idx] + [md_cell, code_cell] + nb['cells'][idx:]
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    print(f'  {nb_name:25s} : inserted Section 1.4 cells at index {idx}–{idx+1}')

print('\nDone.')
