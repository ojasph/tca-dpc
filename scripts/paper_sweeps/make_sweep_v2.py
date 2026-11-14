"""v2, baselines at gamma = 2.5 cm on the 3 layouts: prior + verification (M = 0), equal sampler evaluations (M = 0, B = 24),
DPCC-C, DPCC-T and Diffuser (no constraint handling).
Usage (from dpcc/): python scripts/tca_dpc/paper_sweeps/make_sweep_v2.py   -> config/tca_dpc/sweeps/v2"""
from common import LAY, cfg, write

PRIOR = dict(n_langevin=0, extra_mala=0)
tilt, single = {}, {}
for l in LAY:
    tilt[f'priorcheck_{l}'] = cfg(l, tilt=PRIOR)
    tilt[f'eqcomp24_{l}'] = cfg(l, tilt=PRIOR, top=dict(plan_batch_size=24))
    for m in ['c', 't']: single[f'dpcc{m}_{l}_buf2.5'] = cfg(l, variant=f'dpcc-{m}')
    single[f'diffuser_{l}'] = cfg(l, variant='diffuser')
write('config/tca_dpc/sweeps/v2', tilt, single)
