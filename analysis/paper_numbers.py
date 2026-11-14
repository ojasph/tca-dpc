"""Numbers for the paper's Simulations section (v11s/v12s/v14s/v15s). Usage: python analysis/paper_numbers.py"""
import csv, glob, re, collections, numpy as np, sys
sys.path.insert(0, 'analysis'); from tilt_stats_lib import load
R = collections.defaultdict(list)
for r in csv.DictReader(open('analysis/rows_paper.tsv'), delimiter='\t'):
    R[(r['sweep'], re.sub(r'_s\d$', '', r['job']))].append(r)
def agg(sw, job):
    rows = R[(sw, job)]; out = {'n': len(rows)}
    for k in ['G', 'GC', 'steps', 'viol', 'time', 'f', 'fb']:
        v = [float(r[k]) for r in rows if r.get(k)]; out[k] = np.mean(v) if v else np.nan
    tv = [list(map(float, r['tilt_viol'].split(','))) for r in rows if r.get('tilt_viol')]
    if tv: a = np.array(tv); out['viol_tilt'] = (a[:, 0] * a[:, 1]).sum(); out['viol_fb'] = (a[:, 2] * a[:, 3]).sum(); out['n_fb'] = a[:, 2].sum(); out['n_tilt'] = a[:, 0].sum()
    return out
ours = {'tl': ('sweep_v17s', 'split_tl_buf2.5'), 'tr': ('sweep_v17s', 'split_tr_buf2.5'), 'bh': ('sweep_v17s', 'split_bh_buf2.5')}   # final method: split check (v17)
rows = {'Prior': {l: ('sweep_v14s', f'diffuser_{l}') for l in ours}, 'Prior + verification': {l: ('sweep_v14s', f'priorcheck_{l}') for l in ours},
        'Equal compute B24': {l: ('sweep_v14s', f'eqcomp24_{l}') for l in ours}, 'Fixed beta': {'bh': ('sweep_v14s', 'fixedbeta1e3_bh')},
        'Ours': ours, 'DPCC-C': {l: ('sweep_v14s', f'dpccc_{l}_buf2.5') for l in ours}, 'DPCC-T': {l: ('sweep_v14s', f'dpcct_{l}_buf2.5') for l in ours}}
print('== Table main (2.5 cm): G GC per tl tr bh | mean viol steps time fb (n per layout)')
for name, d in rows.items():
    A = {l: agg(*d[l]) for l in d}
    s = ' '.join(f"{l}:{A[l]['G']:.2f}/{A[l]['GC']:.2f}(n{A[l]['n']})" for l in ['tl', 'tr', 'bh'] if l in A)
    m = lambda k: np.nanmean([A[l][k] for l in A])
    print(f"{name:22s} {s} | viol {m('viol'):.2f} steps {m('steps'):.1f} time {m('time'):.2f} fb {m('fb'):.2f} f {m('f'):.2f}")
print('== Ablation (both, 2.5 cm): GC f fb time n')
abl = [('ours', 'sweep_v11s', 'combo_b'), ('M0', 'sweep_v14s', 'priorcheck_bh')] + [(f'M{m}', 'sweep_v14s', f'M{m}_bh') for m in [1, 2, 10]] + \
      [('kappa0', 'sweep_v14s', 'fixedbeta1e3_bh')] + [(j.split('_')[0], 'sweep_v14s', j) for j in ['kappa100_bh', 'kappa2000_bh', 'delta0.05_bh', 'delta0.5_bh', 'betamax1e5_bh', 'tol1e-2_bh', 'eqcomp24_bh']]
for n, sw, j in abl:
    A = agg(sw, j); print(f"{n:12s} {A['GC']:.2f} {A['f']:.2f} {A['fb']:.2f} {A['time']:.2f} n{A['n']}")
print('== Buffer sweep GC: tilt / DPCC-C / DPCC-T')
for l in ['tl', 'tr']:
    for b in ['0', '0.5', '1', '1.5', '2', '2.5']:
        t = agg(*(ours[l] if b == '2.5' else ('sweep_v14s' if (l == 'tl' or b == '0') else 'sweep_v15s', f'tilt_{l}_buf{b}')))
        sw = 'sweep_v14s' if (l == 'tl' or b in ('0', '2.5')) else 'sweep_v15s'
        c, tt = agg(sw, f'dpccc_{l}_buf{b}'), agg(sw, f'dpcct_{l}_buf{b}')
        print(f"{l} {b:4s} {t['GC']:.2f}(n{t['n']}) {c['GC']:.2f}(n{c['n']}) {tt['GC']:.2f}(n{tt['n']})  time {t['time']:.2f}/{c['time']:.2f}")
print('== Q4 / Q3 from ours')
tot = collections.Counter()
for l, (sw, j) in ours.items():
    A = agg(sw, j); print(l, {k: A.get(k) for k in ['viol_tilt', 'viol_fb', 'n_fb', 'n_tilt', 'fb']})
    T, hp = load(f'analysis/stats_{sw}/{j}_s*') if False else (sum([load(p)[0] for p in sorted(glob.glob(f'analysis/stats_{sw}/{j}_s?'))], []), None)
    beta = np.concatenate([t['beta'] for t in T]); bn = np.concatenate([t['beta_next'] for t in T]); fb = np.concatenate([t['fallback'] for t in T]).astype(bool)
    vmin = np.concatenate([t['vmax_min'] for t in T])
    print(f"   trials {len(T)} beta<1e4 {np.mean(beta < 1e4 - 1):.3f} beta decreases {np.mean(bn < beta):.3f} median beta {np.median(beta):.0f} "
          f"fb {fb.mean():.3f} median vmax_min on fb {np.median(vmin[fb]):.4f} mean {np.mean(vmin[fb]):.4f}")
