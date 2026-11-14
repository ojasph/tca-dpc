"""
    Sweep v17 (2026-11-11): final method `split` (see make_sweep_v16.py) at gamma = 2.5 cm on all 3 layouts, for Table 1,
    Fig. paths and Fig. beta. Submit with DPCC_SAVE_PATHS=1. Usage: python scripts/tca_dpc/paper_sweeps/make_sweep_v17.py [out_dir]
"""
import os
import sys
import copy
import yaml

out = sys.argv[1] if len(sys.argv) > 1 else 'config/tca_dpc/sweeps/sweep_v17s'
base = yaml.safe_load(open('config/tca_dpc/base.yaml'))
LAY = {'bh': 'both-hard', 'tl': 'top-left-hard', 'tr': 'top-right-hard'}
os.makedirs(out, exist_ok=True); names = []
for l, lay in LAY.items():
    for s in range(5):
        c = copy.deepcopy(base)
        c.update(avoiding_halfspace_variants=[lay], seeds=[s], n_trials=10, projection_variants=['tilt-tightened'])
        c['enlarge_constraints'] = {'avoiding': 0.025}
        c['tilt'].update(feas_tol=1e-5, bounds_tol=3e-3)
        j = f'split_{l}_buf2.5_s{s}'; yaml.safe_dump(c, open(f'{out}/{j}.yaml', 'w')); names.append(j)
open(f'{out}/jobs.txt', 'w').write('\n'.join(names) + '\n')
print(f'{len(names)} tasks -> {out}')
