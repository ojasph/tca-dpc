"""v3, buffer sweeps: TCA-DPC vs DPCC-C / DPCC-T for gamma = 0-2 cm on the 3 layouts (gamma = 2.5 cm is in v1 / v2).
Usage (from dpcc/): python scripts/tca_dpc/paper_sweeps/make_sweep_v3.py   -> config/tca_dpc/sweeps/v3"""
from common import LAY, cfg, write

tilt, single = {}, {}
for l in LAY:
    for b in ['0', '0.5', '1', '1.5', '2']:
        tilt[f'split_{l}_buf{b}'] = cfg(l, b)
        for m in ['c', 't']: single[f'dpcc{m}_{l}_buf{b}'] = cfg(l, b, variant=f'dpcc-{m}')
write('config/tca_dpc/sweeps/v3', tilt, single)
