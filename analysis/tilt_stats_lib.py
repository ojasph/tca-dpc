"""Load per-step tilt stats (eval.py *_tilt_stats.npz) into a list of per-trial dicts."""
import glob, json, numpy as np
def load(root):
    """root = analysis/stats_<sweep>/<job>; returns (trials, hparams). Each trial: dict key -> 1D array over control steps, plus 'seed'."""
    T, hp = [], None
    for f in sorted(glob.glob(f'{root}/**/*_tilt_stats.npz', recursive=True)):
        d = np.load(f, allow_pickle=True); hp = json.loads(str(d['hparams_json']))
        seed = int(f.split('GaussianDiffusion/')[1].split('/')[0]); per = {}
        for k in d.files:
            if '_trial' in k:
                n, i = k.rsplit('_trial', 1); per.setdefault(int(i), {})[n] = np.asarray(d[k], float)
        for i in sorted(per): per[i]['seed'] = seed; per[i]['trial'] = i; T.append(per[i])
    return T, hp
def pad(T, key):
    L = max(len(t[key]) for t in T); M = np.full((len(T), L), np.nan)
    for i, t in enumerate(T): M[i, :len(t[key])] = t[key]
    return M
