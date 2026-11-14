"""
    Sweep v15 (2026-11-09): top-right buffer sweep with combo_b (0.5-2 cm; 0 / 2.5 cm in v14 / v12s) + DPCC-C/T.
    Same config builder as make_sweep_v14.py. Usage: python scripts/tca_dpc/paper_sweeps/make_sweep_v15.py [out_dir]
"""
import os
import sys
import copy
import yaml

out = sys.argv[1] if len(sys.argv) > 1 else 'config/tca_dpc/sweeps/sweep_v15s'
base = yaml.safe_load(open('config/tca_dpc/base.yaml'))
BUF = {'0.5': 0.005, '1': 0.01, '1.5': 0.015, '2': 0.02}

def cfg(buf, variant='tilt-tightened', seeds=(0,)):
    c = copy.deepcopy(base)
    c.update(avoiding_halfspace_variants=['top-right-hard'], seeds=list(seeds), n_trials=10)
    c['enlarge_constraints'] = {'avoiding': BUF[buf]}
    c['projection_variants'] = [variant]
    return c

os.makedirs(out, exist_ok=True); names = []
for b in BUF:
    for s in range(5):
        n = f'tilt_tr_buf{b}_s{s}'; yaml.safe_dump(cfg(b, seeds=[s]), open(f'{out}/{n}.yaml', 'w')); names.append(n)
for b in BUF:
    for m in ['c', 't']:
        n = f'dpcc{m}_tr_buf{b}'; yaml.safe_dump(cfg(b, f'dpcc-{m}-tightened', range(5)), open(f'{out}/{n}.yaml', 'w')); names.append(n)
open(f'{out}/jobs.txt', 'w').write('\n'.join(names) + '\n')
print(f'{len(names)} tasks -> {out}')
