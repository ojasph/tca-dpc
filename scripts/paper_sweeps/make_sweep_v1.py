"""v1, main results: TCA-DPC at gamma = 2.5 cm on the 3 layouts.
Usage (from dpcc/): python scripts/tca_dpc/paper_sweeps/make_sweep_v1.py   -> config/tca_dpc/sweeps/v1"""
from common import LAY, cfg, write

write('config/tca_dpc/sweeps/v1', {f'split_{l}_buf2.5': cfg(l) for l in LAY})
