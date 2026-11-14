"""Parse the sweep logs (logs_triton/*.out, written by scripts/run_sweep.sbatch) into one row per (sweep, job, model seed).
Usage (from dpcc/): python analysis/parse_logs_paper.py > analysis/rows_paper.tsv"""
import glob, re, os
pat = {k: re.compile(v) for k, v in dict(
    G=r'^Success rate: ([\d.]+)', GC=r'^Success rate \(goal and constraints\): ([\d.]+)', steps=r'^Avg number of steps: ([\d.]+)',
    viol=r'^Avg number of constraint violations: ([\d.]+)', time=r'^Average computation time per step: ([\d.]+)',
    f=r'mean failing fraction ([\d.]+)', fb=r'fallback rate ([\d.]+), mean beta', beta=r'mean beta ([\d.]+)',
    tilt_viol=r'tilt steps (\d+) \(violation rate ([\d.]+)\), fallback steps (\d+) \(violation rate ([\d.]+)\)').items()}
seen = {}
for fn in sorted(glob.glob('logs_triton/*.out'), key=os.path.getmtime):
    txt = open(fn, errors='ignore').read()
    m = re.search(r'logs_sweep/([^/]+)/([^/]+)/avoiding', txt)
    if not m or 'exit=' not in txt: continue
    sw, job = m.groups()
    for blk in re.split(r'-{10,}Running ', txt)[1:]:
        h = re.match(r'avoiding-d3il - (\S+) - (\S+) \((\d+)\)', blk)
        if not h: continue
        r = dict(sweep=sw, job=job, layout=h[1], variant=h[2], seed=h[3])
        for line in blk.splitlines():
            for k, p in pat.items():
                q = p.search(line)
                if q: r[k] = ','.join(q.groups())
        if 'GC' in r: seen[(sw, job, h[3])] = r
keys = ['sweep', 'job', 'layout', 'variant', 'seed', 'G', 'GC', 'steps', 'viol', 'time', 'f', 'fb', 'beta', 'tilt_viol']
print('\t'.join(keys))
for r in seen.values(): print('\t'.join(r.get(k, '') for k in keys))
