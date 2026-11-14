"""v4, ablations on both-hard at gamma = 2.5 cm (reference: v1 split_bh_buf2.5): fixed beta, number of MALA steps M,
beta update (kappa, delta, beta_max) and a single loose verification tolerance (1e-2 on all constraints).
Usage (from dpcc/): python scripts/tca_dpc/paper_sweeps/make_sweep_v4.py   -> config/tca_dpc/sweeps/v4"""
from common import cfg, write

ABL = {'fixedbeta1e3': dict(kappa=0.0, beta_init=1000.0), 'betamax1e5': dict(beta_max=1e5),
       'tol1e-2': dict(feas_tol=1e-2, bounds_tol=None)}
ABL.update({f'M{M}': dict(n_langevin=M) for M in [1, 2, 10]})
ABL.update({f'kappa{k}': dict(kappa=float(k)) for k in [100, 2000]})
ABL.update({f'delta{d}': dict(delta=d) for d in [0.05, 0.5]})
write('config/tca_dpc/sweeps/v4', {f'{n}_bh': cfg('bh', tilt=t) for n, t in ABL.items()})
