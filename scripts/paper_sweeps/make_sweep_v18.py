"""
    Sweep v18 (2026-11-11): consistency reruns for the final method `split` (feas_tol 1e-5 on safety constraints,
    bounds_tol 3e-3 on input bounds) at gamma = 2.5 cm, same definitions as v14 E2-E4:
      Table 1 baselines (3 layouts): priorcheck (M = 0, no extra MALA), eqcomp24 (M = 0, B = 24)
      Table 2 ablations (both-hard): fixed beta 1e3, M in {1, 2, 10}, kappa in {100, 2000}, delta in {0.05, 0.5}, beta_max 1e5
    The reference (split, 2.5 cm) is v17. One task per (setting, seed).
    Usage: python scripts/tca_dpc/paper_sweeps/make_sweep_v18.py [out_dir] [--smoke]
"""
import os
import sys
import copy
import yaml

args = [a for a in sys.argv[1:] if not a.startswith('--')]
smoke = '--smoke' in sys.argv
out = args[0] if args else ('config/tca_dpc/sweeps/smoke_v18' if smoke else 'config/tca_dpc/sweeps/sweep_v18s')
base = yaml.safe_load(open('config/tca_dpc/base.yaml'))
base['tilt'].update(feas_tol=1e-5, bounds_tol=3e-3)
LAY = {'bh': 'both-hard', 'tl': 'top-left-hard', 'tr': 'top-right-hard'}

def cfg(lay, tilt=None, top=None, seed=0, n_trials=10):
    c = copy.deepcopy(base)
    c.update(avoiding_halfspace_variants=[LAY[lay]], seeds=[seed], n_trials=n_trials, projection_variants=['tilt-tightened'])
    c['enlarge_constraints'] = {'avoiding': 0.025}
    c['tilt'].update(tilt or {}); c.update(top or {})
    return c

jobs = {}
for l in LAY:
    jobs[f'priorcheck_{l}'] = (l, dict(n_langevin=0, extra_mala=0), None)
    jobs[f'eqcomp24_{l}'] = (l, dict(n_langevin=0, extra_mala=0), dict(plan_batch_size=24))
jobs['fixedbeta1e3_bh'] = ('bh', dict(kappa=0.0, beta_init=1000.0), None)
for M in [1, 2, 10]: jobs[f'M{M}_bh'] = ('bh', dict(n_langevin=M), None)
for k in [100, 2000]: jobs[f'kappa{k}_bh'] = ('bh', dict(kappa=float(k)), None)
for d in [0.05, 0.5]: jobs[f'delta{d}_bh'] = ('bh', dict(delta=d), None)
jobs['betamax1e5_bh'] = ('bh', dict(beta_max=1e5), None)

os.makedirs(out, exist_ok=True); names = []
pick = ['priorcheck_bh', 'eqcomp24_bh', 'fixedbeta1e3_bh', 'M10_bh'] if smoke else list(jobs)
for n in pick:
    l, t, top = jobs[n]
    for s in ([0] if smoke else range(5)):
        j = f'{n}_s{s}'; yaml.safe_dump(cfg(l, t, top, s, n_trials=1 if smoke else 10), open(f'{out}/{j}.yaml', 'w')); names.append(j)
open(f'{out}/jobs.txt', 'w').write('\n'.join(names) + '\n')
print(f'{len(names)} tasks -> {out}')
