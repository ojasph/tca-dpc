"""Shared config builder for the paper sweeps. All settings start from the final method, config/tca_dpc/tca_dpc.yaml.
Tilt settings run one task per model seed (<name>_s<k>); DPCC / Diffuser run all 5 seeds in one task (they are fast)."""
import os
import copy
import yaml

BASE = yaml.safe_load(open('config/tca_dpc/tca_dpc.yaml'))
LAY = {'tl': 'top-left-hard', 'tr': 'top-right-hard', 'bh': 'both-hard'}
BUF = {'0': 0.0, '0.5': 0.005, '1': 0.01, '1.5': 0.015, '2': 0.02, '2.5': 0.025}    # constraint buffer gamma (m)
SEEDS = range(5)


def cfg(lay, buf='2.5', variant='tilt', tilt=None, top=None):
    """variant: 'tilt' (TCA-DPC), 'dpcc-c', 'dpcc-t' or 'diffuser'; '-tightened' is added when buf > 0."""
    c = copy.deepcopy(BASE)
    c.update(avoiding_halfspace_variants=[LAY[lay]], n_trials=10)
    c['enlarge_constraints'] = {'avoiding': BUF[buf]}
    c['projection_variants'] = [variant + ('-tightened' if buf != '0' and variant != 'diffuser' else '')]
    c['tilt'].update(tilt or {}); c.update(top or {})
    return c


def write(out, tilt_jobs, single_jobs=None):
    """tilt_jobs / single_jobs: {name: config}. Writes <out>/<task>.yaml and <out>/jobs.txt (one task per line)."""
    os.makedirs(out, exist_ok=True); names = []
    for n, c in tilt_jobs.items():
        for s in SEEDS:
            names.append(f'{n}_s{s}'); yaml.safe_dump(dict(c, seeds=[s]), open(f'{out}/{names[-1]}.yaml', 'w'))
    for n, c in (single_jobs or {}).items():
        names.append(n); yaml.safe_dump(dict(c, seeds=list(SEEDS)), open(f'{out}/{n}.yaml', 'w'))
    open(f'{out}/jobs.txt', 'w').write('\n'.join(names) + '\n')
    print(f'{len(names)} tasks -> {out}')
