"""Numbers for the paper's tables from the v1-v4 sweeps (scripts/paper_sweeps/), averaged over the 5 model seeds.
Needs analysis/rows_paper.tsv (python analysis/parse_logs_paper.py > analysis/rows_paper.tsv).
GC = success rate (goal reached and constraints satisfied), time = s per control step, f = failing fraction, fb = fallback rate.
Usage (from dpcc/): python analysis/paper_numbers.py"""
import csv, re, collections, numpy as np

R = collections.defaultdict(list)
for r in csv.DictReader(open('analysis/rows_paper.tsv'), delimiter='\t'):
    R[(r['sweep'], re.sub(r'_s\d$', '', r['job']))].append(r)
LAY = ['tl', 'tr', 'bh']


def agg(sw, job):
    rows = R[(sw, job)]; out = {'n': len(rows)}
    for k in ['G', 'GC', 'viol', 'time', 'f', 'fb']:
        v = [float(r[k]) for r in rows if r.get(k)]; out[k] = np.mean(v) if v else np.nan
    return out


def line(name, cells):
    A = [agg(*c) for c in cells]
    gc = ' '.join(f"{a['GC']:.2f}(n{a['n']})" for a in A)
    print(f"{name:24s} {gc} | time {np.nanmean([a['time'] for a in A]):.2f} fb {np.nanmean([a['fb'] for a in A]):.2f}")


print('== Main results and baselines, gamma = 2.5 cm: GC top-left top-right both')
line('TCA-DPC (v1)', [('v1', f'split_{l}_buf2.5') for l in LAY])
for name, job in [('Prior + verification', 'priorcheck_{}'), ('Equal evaluations B24', 'eqcomp24_{}'),
                  ('DPCC-C', 'dpccc_{}_buf2.5'), ('DPCC-T', 'dpcct_{}_buf2.5'), ('Diffuser', 'diffuser_{}')]:
    line(name, [('v2', job.format(l)) for l in LAY])

print('== Buffer sweep (v3; 2.5 cm from v1/v2): GC TCA-DPC / DPCC-C / DPCC-T')
for l in LAY:
    for b in ['0', '0.5', '1', '1.5', '2', '2.5']:
        sw, sb = ('v1', 'v2') if b == '2.5' else ('v3', 'v3')
        t, c, tt = agg(sw, f'split_{l}_buf{b}'), agg(sb, f'dpccc_{l}_buf{b}'), agg(sb, f'dpcct_{l}_buf{b}')
        print(f"{l} {b:4s} {t['GC']:.2f} / {c['GC']:.2f} / {tt['GC']:.2f}")

print('== Ablations, both-hard, gamma = 2.5 cm (v4): GC f fb time')
for n, (sw, j) in {'reference': ('v1', 'split_bh_buf2.5'),
                   **{a: ('v4', f'{a}_bh') for a in ['M1', 'M2', 'M10', 'fixedbeta1e3', 'kappa100', 'kappa2000',
                                                    'delta0.05', 'delta0.5', 'betamax1e5', 'tol1e-2']}}.items():
    A = agg(sw, j); print(f"{n:14s} {A['GC']:.2f} {A['f']:.2f} {A['fb']:.2f} {A['time']:.2f} n{A['n']}")
