from pathlib import Path
import pandas as pd
from viz_analysis import compute_marker_csv

SEEDS = [0, 1, 2, 3, 4]

CONFIGS = [
    {'label': 'AdamW fast',       'save_root': 'runs/phase2_optim/adamw_fast_fixedkf',      'family': 'AdamW', 'momentum': 0.9,  'regime': 'fast'},
    {'label': 'AdamW slow',       'save_root': 'runs/phase2_optim/adamw_slow_fixedkf',      'family': 'AdamW', 'momentum': 0.9,  'regime': 'slow'},
    {'label': 'EGD m=0 fast',     'save_root': 'runs/phase2_v2_flexkf/egd_m0_fast',         'family': 'EGD',   'momentum': 0.0,  'regime': 'fast'},
    {'label': 'EGD m=0 slow',     'save_root': 'runs/phase2_optim/egd_m0_slow_fixedkf',     'family': 'EGD',   'momentum': 0.0,  'regime': 'slow'},
    {'label': 'EGD m=0.9 fast',   'save_root': 'runs/phase2_optim/egd_m0.9_fast_fixedkf',   'family': 'EGD',   'momentum': 0.9,  'regime': 'fast'},
    {'label': 'EGD m=0.9 slow',   'save_root': 'runs/phase2_optim/egd_m0.9_slow_fixedkf',   'family': 'EGD',   'momentum': 0.9,  'regime': 'slow'},
    {'label': 'Muon m=0 fast',    'save_root': 'runs/phase2_v2_flexkf/muon_m0_fast',        'family': 'Muon',  'momentum': 0.0,  'regime': 'fast'},
    {'label': 'Muon m=0 slow',    'save_root': 'runs/phase2_optim/muon_m0_slow_fixedkf',    'family': 'Muon',  'momentum': 0.0,  'regime': 'slow'},
    {'label': 'Muon m=0.95 fast', 'save_root': 'runs/phase2_optim/muon_m0.95_fast_fixedkf', 'family': 'Muon',  'momentum': 0.95, 'regime': 'fast'},
    {'label': 'Muon m=0.95 slow', 'save_root': 'runs/phase2_optim/muon_m0.95_slow_fixedkf', 'family': 'Muon',  'momentum': 0.95, 'regime': 'slow'},
]

CSV_PATH = Path('analysis_csv/analysis_markers.csv')


if __name__ == '__main__':
    df = compute_marker_csv(CONFIGS, SEEDS, CSV_PATH)
    print(f'wrote {CSV_PATH}  ({len(df)} rows)')

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 50)
    agg = df.groupby('config', sort=False).agg(
        T_mem=('T_mem', 'median'),
        T_circuit=('T_circuit', 'median'),
        T_grok=('T_grok', 'median'),
        gap=('gap', 'median'),
        gap_ratio=('gap_ratio', 'median'),
        circuit_lead=('circuit_lead', 'median'),
        T_exc_min=('T_exc_min', 'median'),
        exc_min=('exc_min', 'median'),
        T_exc_recover=('T_exc_recover', 'median'),
        cleanup_dur=('cleanup_dur', 'median'),
        n_keys=('n_keys', 'mean'),
        l2_peak_ep=('T_l2_peak', 'median'),
        gini_final=('gini_final', 'median'),
    )
    print('=== Median per config ===')
    print(agg.round(2))

    print('\n=== Per-seed std (timing variability) ===')
    std = df.groupby('config', sort=False)[['T_mem', 'T_circuit', 'T_grok', 'gap']].std()
    print(std.round(1))
