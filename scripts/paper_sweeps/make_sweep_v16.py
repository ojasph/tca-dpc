"""
    Sweep v16 (2026-11-10): beat DPCC across buffers. Split the verification tolerance: strict (feas_tol 1e-5) on the safety
    constraints (halfspaces, obstacles), loose (bounds_tol 3e-3) on the velocity box bounds only.
    Evidence (rows_paper.tsv tilt_viol split, stats npz):
      - strict check (old v6-v10 config): accepted plans ~never violate (0-0.2%/step); ~100% of violations come from the
        fallback, used on ~53% of steps, and 93-96% of those fallbacks had a best candidate within 3e-3 (velocity overshoot).
      - loose check (combo_b, tol 3e-3 everywhere): fallback only 1-7% of steps, but accepted plans violate at 1-7%/step
        at small buffers (tl 0-1 cm, bh 0 cm); tol 1e-2 at bh 2.5 cm -> 0.04 success.
      - remaining hard steps (combo_b, top-right): best candidate > 3e-3 off, previous-plan fallback violates 26-34%/step.
      - beta rule (kappa, delta, beta_max, fixed beta; v14s at bh 2.5 cm): all within noise -> not varied here.
    Settings (all on combo_b otherwise):
      split       split tolerance, DPCC-free (fallback: previous plan <= 3 times, then least-violating candidate)
      split_esc   split + escalation: when no candidate passes after the extra MALA steps, up to 3 rounds of B fresh noises
                  with M MALA steps each, before the fallback (projection-free; extra cost only on stuck steps)
      split_m3    split + penalty margin 3e-3 (hardest cells only)
    Plus DPCC-T on both-hard 0.5-2 cm (missing). One task per (setting, layout, buffer, seed).
    Usage: python scripts/tca_dpc/paper_sweeps/make_sweep_v16.py [out_dir] [--smoke]
"""
import os
import sys
import copy
import yaml

args = [a for a in sys.argv[1:] if not a.startswith('--')]
smoke = '--smoke' in sys.argv
out = args[0] if args else ('config/tca_dpc/sweeps/smoke_v16' if smoke else 'config/tca_dpc/sweeps/sweep_v16s')
base = yaml.safe_load(open('config/tca_dpc/base.yaml'))
LAY = {'bh': 'both-hard', 'tl': 'top-left-hard', 'tr': 'top-right-hard'}
BUF = {'0': 0.0, '0.5': 0.005, '1': 0.01, '1.5': 0.015, '2': 0.02}
SPLIT = dict(feas_tol=1e-5, bounds_tol=3e-3)
SETTINGS = {'split': SPLIT, 'split_esc': dict(SPLIT, escalate=3), 'split_m3': dict(SPLIT, margin=3e-3)}
CELLS = {'split': [(l, b) for l in LAY for b in BUF], 'split_esc': [(l, b) for l in LAY for b in BUF],
         'split_m3': [('tl', '0.5'), ('tr', '0.5'), ('bh', '0')]}

def cfg(lay, buf, tilt=None, seeds=(0,), n_trials=10, variant=None):
    c = copy.deepcopy(base)
    c.update(avoiding_halfspace_variants=[LAY[lay]], seeds=list(seeds), n_trials=n_trials)
    c['enlarge_constraints'] = {'avoiding': BUF[buf]}
    c['projection_variants'] = [variant or ('tilt' if buf == '0' else 'tilt-tightened')]
    c['tilt'].update(tilt or {})
    return c

os.makedirs(out, exist_ok=True); names = []
def put(j, c): yaml.safe_dump(c, open(f'{out}/{j}.yaml', 'w')); names.append(j)
if smoke:   # 1 episode each, both code paths, on the cell where combo_b fails most
    for n in ['split', 'split_esc']: put(f'{n}_tl_buf0.5_s0', cfg('tl', '0.5', SETTINGS[n], n_trials=1))
else:
    for n, cells in CELLS.items():
        for l, b in cells:
            for s in range(5): put(f'{n}_{l}_buf{b}_s{s}', cfg(l, b, SETTINGS[n], seeds=[s]))
    for b in ['0.5', '1', '1.5', '2']:      # DPCC-T reference on both-hard (all 5 seeds in one task, ~0.5 s/step)
        put(f'dpcct_bh_buf{b}', cfg('bh', b, seeds=range(5), variant='dpcc-t-tightened'))
open(f'{out}/jobs.txt', 'w').write('\n'.join(names) + '\n')
print(f'{len(names)} tasks -> {out}')
