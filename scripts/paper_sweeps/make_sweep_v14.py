"""
    Sweep v14 (2026-11-09): paper experiments with the final setting combo_b (config/tca_dpc/base.yaml).
    E1 buffer sweep (top-left 0-2 cm; both-hard / top-right 0 cm) + DPCC-C/T; E2 baselines (prior + check, equal-compute prior B24,
    fixed beta 1e3, Diffuser); E3 M in {1, 2, 10}; E4 kappa / delta / beta_max; E5 tol 1e-2.
    Tilt settings: one task per seed (<name>_s<k>); DPCC / Diffuser: all 5 seeds in one task.
    Usage: python scripts/tca_dpc/paper_sweeps/make_sweep_v14.py [out_dir]
"""
import os
import sys
import copy
import yaml

out = sys.argv[1] if len(sys.argv) > 1 else 'config/tca_dpc/sweeps/sweep_v14s'
base = yaml.safe_load(open('config/tca_dpc/base.yaml'))
LAY = {'bh': 'both-hard', 'tl': 'top-left-hard', 'tr': 'top-right-hard'}
BUF = {'0': 0.0, '0.5': 0.005, '1': 0.01, '1.5': 0.015, '2': 0.02, '2.5': 0.025}

def cfg(lay, buf='2.5', variant=None, tilt=None, top=None, seeds=(0,)):
    c = copy.deepcopy(base)
    c.update(avoiding_halfspace_variants=[LAY[lay]], seeds=list(seeds), n_trials=10)
    c['enlarge_constraints'] = {'avoiding': BUF[buf]}
    c['projection_variants'] = [variant or ('tilt' if buf == '0' else 'tilt-tightened')]
    c['tilt'].update(tilt or {}); c.update(top or {})
    return c

tilt_jobs, single_jobs = {}, {}
# E1 tilt
for b in ['0', '0.5', '1', '1.5', '2']: tilt_jobs[f'tilt_tl_buf{b}'] = cfg('tl', b)
for l in ['bh', 'tr']: tilt_jobs[f'tilt_{l}_buf0'] = cfg(l, '0')
# E1 DPCC (all seeds in one task)
for l, bufs in [('tl', BUF), ('bh', ['0', '2.5']), ('tr', ['0', '2.5'])]:
    for b in bufs:
        for m in ['c', 't']:
            single_jobs[f'dpcc{m}_{l}_buf{b}'] = cfg(l, b, variant=f'dpcc-{m}' + ('' if b == '0' else '-tightened'), seeds=range(5))
# E2
for l in LAY:
    tilt_jobs[f'priorcheck_{l}'] = cfg(l, tilt=dict(n_langevin=0, extra_mala=0))
    tilt_jobs[f'eqcomp24_{l}'] = cfg(l, tilt=dict(n_langevin=0, extra_mala=0), top=dict(plan_batch_size=24))
    single_jobs[f'diffuser_{l}'] = cfg(l, variant='diffuser', seeds=range(5))
tilt_jobs['fixedbeta1e3_bh'] = cfg('bh', tilt=dict(kappa=0.0, beta_init=1000.0))
# E3
for M in [1, 2, 10]: tilt_jobs[f'M{M}_bh'] = cfg('bh', tilt=dict(n_langevin=M))
# E4
for k in [100, 2000]: tilt_jobs[f'kappa{k}_bh'] = cfg('bh', tilt=dict(kappa=float(k)))
for d in [0.05, 0.5]: tilt_jobs[f'delta{d}_bh'] = cfg('bh', tilt=dict(delta=d))
tilt_jobs['betamax1e5_bh'] = cfg('bh', tilt=dict(beta_max=1e5))
# E5
tilt_jobs['tol1e-2_bh'] = cfg('bh', tilt=dict(feas_tol=1e-2))

os.makedirs(out, exist_ok=True); names = []
for n, c in tilt_jobs.items():
    for s in range(5):
        c2 = dict(c, seeds=[s]); yaml.safe_dump(c2, open(f'{out}/{n}_s{s}.yaml', 'w')); names.append(f'{n}_s{s}')
for n, c in single_jobs.items():
    yaml.safe_dump(c, open(f'{out}/{n}.yaml', 'w')); names.append(n)
open(f'{out}/jobs.txt', 'w').write('\n'.join(names) + '\n')
print(f'{len(tilt_jobs)} tilt settings x 5 seeds + {len(single_jobs)} DPCC/Diffuser tasks = {len(names)} tasks -> {out}')
