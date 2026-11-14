"""Paper figures from the final-setting runs (v11s/v12s combo_b, v14s, v15s).
Needs analysis/rows_paper.tsv (scripts/parse_logs_paper.py on Triton) and analysis/stats_<sweep>/ (stats .npz + paths .pkl).
Usage (from dpcc/): python analysis/plot_paper.py   -> analysis/figs/paper_*.{pdf,png}"""
import csv, glob, pickle, re, sys, collections, numpy as np, yaml, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
sys.path.insert(0, 'analysis'); from tilt_stats_lib import load, pad

OUT = 'analysis/figs'
plt.rcParams.update({  # paper style (L4DC / PMLR body font is Computer Modern)
    'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False, 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.5, 'axes.edgecolor': 'black',
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'axes.grid': False, 'legend.frameon': False, 'lines.linewidth': 1.0, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42})
# fixed colour per method (categorical slots 1-5 + neutral for the prior), never re-assigned
C = {'Ours': '#2a78d6', 'DPCC-C': '#eb6834', 'DPCC-T': '#1baf7a', 'Prior + verification': '#eda100',
     'Equal sampler evaluations': '#6250d6', 'Prior': '#8a8984'}
DISP = {'Ours': 'TCA-DPC'}   # display name in legends
MK = {'Ours': 'o', 'DPCC-C': 's', 'DPCC-T': '^', 'Prior + verification': 'D', 'Equal sampler evaluations': 'v', 'Prior': 'x'}
LAY = {'tl': 'Top-left', 'tr': 'Top-right', 'bh': 'Both'}
OURS = {'tl': ('sweep_v17s', 'split_tl_buf2.5'), 'tr': ('sweep_v17s', 'split_tr_buf2.5'), 'bh': ('sweep_v17s', 'split_bh_buf2.5')}   # final method: split check (v17)

def save(fig, name):
    for d in (OUT, 'Constraint_Aware_Diffusion_Predictive_Control/figs'):
        for ext in ('pdf', 'png'): fig.savefig(f'{d}/{name}.{ext}', dpi=200)
    plt.close(fig); print('wrote', name)

# ---------------- per-seed rows ----------------
R = collections.defaultdict(list)
for r in csv.DictReader(open('analysis/rows_paper.tsv'), delimiter='\t'):
    R[(r['sweep'], re.sub(r'_s\d$', '', r['job']))].append(r)
def vals(sw, job, k='GC'): return np.array([float(r[k]) for r in R[(sw, job)] if r.get(k)])
def ms(sw, job, k='GC'):
    v = vals(sw, job, k); return (v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0., len(v)) if len(v) else (np.nan, 0., 0)
def src(method, l, b='2.5'):
    if method == 'Ours': return OURS[l] if b == '2.5' else ('sweep_v15s' if (l == 'tr' and b != '0') else 'sweep_v14s', f'tilt_{l}_buf{b}')
    sw = 'sweep_v15s' if (l == 'tr' and b not in ('0', '2.5')) else 'sweep_v14s'
    return sw, {'DPCC-C': f'dpccc_{l}_buf{b}', 'DPCC-T': f'dpcct_{l}_buf{b}', 'Prior': f'diffuser_{l}',
                'Prior + verification': f'priorcheck_{l}', 'Equal sampler evaluations': f'eqcomp24_{l}'}[method]

# ---------------- Fig: buffer sweep, top-left, v10 tilt setting ----------------
# tilt: strict check (tol 1e-5), beta0 1e3, kappa 3e3, delta 0.05, beta_max 1e4, M5 B4 (identical config in v6 / v10 / v9)
# DPCC-C / DPCC-T: v14s (same frozen models)
# both-hard: tilt 0 cm = v8 M5_B4, 2.5 cm = v7 nd_cap3_extra10_tight; DPCC-C 0.5-2 cm = v10 dpcc_bh (dpcc-c-tightened);
# DPCC-T has no both-hard runs at 0.5-2 cm (markers only at 0 / 2.5)
BUFS = ['0', '0.5', '1', '1.5', '2', '2.5']
MID = ['0.5', '1', '1.5', '2']
OURS_BUF = {'tl': {'0': ('sweep_v6', 'tl_nd_extra10'), '2.5': ('sweep_v9', 'nd_cap3_extra10_tight_tl'),
                   **{b: ('sweep_v10', f'tilt_tl_buf{b}cm') for b in MID}},
            'bh': {'0': ('sweep_v8', 'M5_B4'), '2.5': ('sweep_v7', 'nd_cap3_extra10_tight'),
                   **{b: ('sweep_v10', f'tilt_bh_buf{b}cm') for b in MID}}}
def buf_src(m, l, b):
    if m == 'Ours': return OURS_BUF[l][b]
    if m == 'DPCC-C' and l == 'bh' and b in MID: return 'sweep_v10', f'dpcc_bh_buf{b}cm'
    return src(m, l, b)
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2), sharey=True)
for ax, l in zip(axes, ['tl', 'bh']):
    for m, dx in [('Ours', -0.03), ('DPCC-C', 0.), ('DPCC-T', 0.03)]:
        y = np.array([ms(*buf_src(m, l, b)) for b in BUFS])
        x = np.array([float(b) for b in BUFS]) + dx; ok = ~np.isnan(y[:, 0])
        ax.errorbar(x[ok], y[ok, 0], yerr=y[ok, 1], color=C[m], marker=MK[m], ms=3.5, lw=1.0 if ok.all() else 0,
                    capsize=1.5, elinewidth=0.6, mec='white', mew=0.5, label=DISP.get(m, m))
        print('buffer', l, m, np.round(y[:, 0], 2), 'n', y[:, 2])
    ax.set_title(LAY[l]); ax.set_xlabel(r'Tightening $\gamma$ [cm]'); ax.set_xticks([float(b) for b in BUFS])
axes[0].set_ylim(0.35, 1.03); axes[0].set_ylabel('Success (goal + constraints)')
axes[1].legend(loc='lower right')
save(fig, 'paper_buffer')

# ---------------- Fig: both-hard, tilt vs DPCC-C, 0-2 cm ----------------
B02 = BUFS[:-1]
fig, ax = plt.subplots(figsize=(3.0, 2.2))
for m, dx in [('Ours', -0.02), ('DPCC-C', 0.02)]:
    y = np.array([ms(*buf_src(m, 'bh', b)) for b in B02])
    ax.errorbar(np.array([float(b) for b in B02]) + dx, y[:, 0], yerr=y[:, 1], color=C[m], marker=MK[m], ms=3.5, lw=1.0,
                capsize=1.5, elinewidth=0.6, mec='white', mew=0.5, label={'Ours': 'TCA-DPC', 'DPCC-C': 'DPCC'}[m])
    print('buffer_bh', m, np.round(y[:, 0], 2), 'n', y[:, 2])
ax.set_title(LAY['bh']); ax.set_xlabel(r'Tightening $\gamma$ [cm]'); ax.set_xticks([float(b) for b in B02])
ax.set_ylim(0.3, 1.03); ax.set_ylabel('Success (goal + constraints)'); ax.legend(loc='lower right')
save(fig, 'paper_buffer_bh')

# ---------------- Fig: top-right, combo_b buffer sweep (v14s 0 cm, v15s 0.5-2 cm, v12s 2.5 cm) ----------------
fig, ax = plt.subplots(figsize=(3.0, 2.2))
for m, dx in [('Ours', -0.03), ('DPCC-C', 0.), ('DPCC-T', 0.03)]:
    y = np.array([ms(*src(m, 'tr', b)) for b in BUFS])
    ax.errorbar(np.array([float(b) for b in BUFS]) + dx, y[:, 0], yerr=y[:, 1], color=C[m], marker=MK[m], ms=3.5, lw=1.0,
                capsize=1.5, elinewidth=0.6, mec='white', mew=0.5, label=DISP.get(m, m))
    print('buffer_tr', m, np.round(y[:, 0], 2), 'n', y[:, 2])
ax.set_title(LAY['tr']); ax.set_xlabel(r'Tightening $\gamma$ [cm]'); ax.set_xticks([float(b) for b in BUFS])
ax.set_ylim(0.55, 1.03); ax.set_ylabel('Success (goal + constraints)'); ax.legend(loc='lower right')
save(fig, 'paper_buffer_tr')

# ---------------- Fig: split check (v16s) vs DPCC-C / DPCC-T, 3 layouts, 0-2 cm ----------------
# split = combo_b with feas_tol 1e-5 on safety constraints, bounds_tol 3e-3 on velocity bounds (projection-free)
# rows: analysis/rows_v16.tsv (per-seed). DPCC: v14s/v15s, both-hard 0.5-2 cm DPCC-C from v10; DPCC-T both-hard from v16s once run
for r in csv.DictReader(open('analysis/rows_v16.tsv'), delimiter='\t'):
    R[(r['sweep'], re.sub(r'_s\d$', '', r['job']))].append(r)
def split_src(m, l, b):
    if m == 'Ours': return 'sweep_v16s', f'split_{l}_buf{b}'
    if l == 'bh' and b in MID: return ('sweep_v10', f'dpcc_bh_buf{b}cm') if m == 'DPCC-C' else ('sweep_v16s', f'dpcct_bh_buf{b}')
    return src(m, l, b)
B02 = BUFS[:-1]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2), sharey=True)
for ax, l in zip(axes, ['tl', 'tr', 'bh']):
    for m, dx in [('Ours', -0.03), ('DPCC-C', 0.), ('DPCC-T', 0.03)]:
        y = np.array([ms(*split_src(m, l, b)) for b in B02])
        x = np.array([float(b) for b in B02]) + dx; ok = ~np.isnan(y[:, 0])
        ax.errorbar(x[ok], y[ok, 0], yerr=y[ok, 1], color=C[m], marker=MK[m], ms=3.5, lw=1.0 if ok.all() else 0,
                    capsize=1.5, elinewidth=0.6, mec='white', mew=0.5, label=DISP.get(m, m))
        print('buffer_split', l, m, np.round(y[:, 0], 2), 'n', y[:, 2])
    ax.set_title(LAY[l]); ax.set_xlabel(r'Tightening $\gamma$ [cm]'); ax.set_xticks([float(b) for b in B02])
axes[0].set_ylim(0.3, 1.03); axes[0].set_ylabel('Success (goal + constraints)')
axes[2].legend(loc='lower right')
save(fig, 'paper_buffer_split')

# ---------------- Fig: baselines per layout (2.5 cm) ----------------
meths = ['Prior', 'Prior + verification', 'Equal sampler evaluations', 'Ours', 'DPCC-C']
fig, ax = plt.subplots(figsize=(6.0, 2.1)); w = 0.16
for i, m in enumerate(meths):
    y = np.array([ms(*src(m, l)) for l in ['tl', 'tr', 'bh']])
    xs = np.arange(3) + (i - 2) * (w + 0.012)
    ax.bar(xs, y[:, 0], w, color=C[m], label=DISP.get(m, m), zorder=2, edgecolor='white', linewidth=0.8)
    ax.errorbar(xs, y[:, 0], yerr=y[:, 1], fmt='none', ecolor='black', elinewidth=0.5, capsize=1.5, zorder=3)
ax.set_xticks(range(3)); ax.set_xticklabels([LAY[l] for l in ['tl', 'tr', 'bh']]); ax.grid(axis='x', visible=False)
ax.set_ylabel('Success (goal + constraints)'); ax.set_ylim(0, 1.08)
ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.2), handlelength=1)
save(fig, 'paper_baselines')

# ---------------- Fig: number of MALA steps M (both, 2.5 cm) ----------------
Ms = [(0, ('sweep_v14s', 'priorcheck_bh')), (1, ('sweep_v14s', 'M1_bh')), (2, ('sweep_v14s', 'M2_bh')),
      (5, OURS['bh']), (10, ('sweep_v14s', 'M10_bh'))]
fig, axes = plt.subplots(1, 3, figsize=(6.0, 1.9))
for ax, (k, lab) in zip(axes, [('GC', 'Success (goal + constr.)'), ('f', r'Mean failure fraction $\bar f$'), ('time', 'Time per update [s]')]):
    y = np.array([ms(*s, k=k) for _, s in Ms])
    ax.errorbar([m for m, _ in Ms], y[:, 0], yerr=y[:, 1], color=C['Ours'], marker='o', ms=3.5, lw=1.0, capsize=1.5, elinewidth=0.6, mec='white', mew=0.5)
    ax.set_xlabel('MALA steps $M$'); ax.set_title(lab); ax.set_xticks([0, 1, 2, 5, 10])
    if k == 'GC': ax.set_ylim(0, 1.05)
    if k == 'f': ax.axhline(0.25, color='black', lw=0.5, ls=(0, (4, 2))); ax.text(9.8, 0.247, r'$\delta$', ha='right', va='top', fontsize=7)
n10 = ms(*Ms[-1][1])[2]
if n10 < 5: axes[0].text(10, 0.05, f'M=10: {n10}/5 models', ha='right', fontsize=7, color='#52514e')
save(fig, 'paper_M')

# ---------------- Fig: beta_t and f_t over the episode (ours, 2.5 cm) ----------------
# time aligned to the onset t0 = first step with f_t > 0 (first sampled sequence failing verification); 50 episodes per layout.
# beta_t: median + 10 individual episodes on a log axis (beta = 0 drawn at the bottom, tick "0"; positive beta >= beta0 = 10);
# f_t: mean +- s.e. (f_t takes values k/B, so quantiles are uninformative). Points with < 25 episodes are dropped.
PRE, POST, NMIN = 10, 50, 25
# style 'traces': beta median + 10 episodes (paper_beta); 'iqr': beta median + interquartile range (paper_beta_iqr)
OURS0 = {l: ('sweep_v16s', f'split_{l}_buf0') for l in ['tl', 'tr', 'bh']}     # final method without tightening (v16)
for style, SRC, out in [('traces', OURS0, 'paper_beta'), ('iqr', OURS0, 'paper_beta_iqr'), ('traces', OURS, 'paper_beta_g25'), ('iqr', OURS, 'paper_beta_g25_iqr')]:   # paper: gamma = 0
    fig, axes = plt.subplots(2, 3, figsize=(6.0, 2.7), sharex=True, sharey='row')
    rng = np.random.default_rng(0)
    for j, l in enumerate(['tl', 'tr', 'bh']):
        sw, job = SRC[l]; T = sum([load(p)[0] for p in sorted(glob.glob(f'analysis/stats_{sw}/{job}_s?'))], [])
        on = [int(np.argmax(t['fail_frac'] > 0)) for t in T]
        show = rng.choice(len(T), 10, replace=False)
        for i, key in enumerate(['beta', 'fail_frac']):
            A = np.full((len(T), PRE + POST + 1), np.nan)
            for e, (t, t0) in enumerate(zip(T, on)):
                x = t[key][max(t0 - PRE, 0):t0 + POST + 1]; A[e, PRE - min(t0, PRE):PRE - min(t0, PRE) + len(x)] = x
            n = (~np.isnan(A)).sum(0); keep = n >= NMIN; s_ = np.arange(-PRE, POST + 1); ax = axes[i, j]
            ax.axvline(0, color='#8a8984', lw=0.5, ls=':')
            if key == 'beta':
                B = np.maximum(A, 1.)
                if style == 'traces':
                    for e in show: ax.plot(s_, B[e], color=C['Ours'], lw=0.4, alpha=0.35)
                else:       # interquartile range over episodes
                    q1, q3 = np.nanpercentile(B, 25, 0), np.nanpercentile(B, 75, 0)
                    ax.fill_between(s_[keep], q1[keep], q3[keep], color=C['Ours'], alpha=0.18, lw=0)
                ax.plot(s_[keep], np.nanmedian(B, 0)[keep], color=C['Ours'], lw=1.2)
                ax.axhline(1e4, color='black', lw=0.5, ls=(0, (4, 2)))
                ax.set_yscale('log'); ax.set_ylim(0.7, 2e4); ax.set_yticks([1, 1e1, 1e2, 1e3, 1e4])
                ax.set_yticklabels(['$0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$']); ax.minorticks_off()
            else:
                m = np.nanmean(A, 0)[keep]; se = np.nanstd(A, 0)[keep] / np.sqrt(n[keep])
                ax.fill_between(s_[keep], m - se, m + se, color=C['Ours'], alpha=0.18, lw=0); ax.plot(s_[keep], m, color=C['Ours'], lw=1.0)
                ax.axhline(0.25, color='black', lw=0.5, ls=(0, (4, 2))); ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            print('beta fig', l, key, 'last kept s =', s_[keep][-1])
        ax.set_xlim(-PRE, POST)
        axes[0, j].set_title(f'({"abc"[j]}) {LAY[l]}', pad=3); axes[1, j].set_xlabel(r'Steps since first failure $t-t_0$')
    axes[0, 0].set_ylabel(r'$\beta_t$'); axes[1, 0].set_ylabel(r'$f_t$')
    axes[0, 0].text(0.03, 1e4 * 0.55, r'$\beta_{\max}$', transform=axes[0, 0].get_yaxis_transform(), va='top', fontsize=7)
    axes[1, 0].text(0.03, 0.27, r'$\delta$', transform=axes[1, 0].get_yaxis_transform(), va='bottom', fontsize=7)
    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    save(fig, out)

# ---------------- Fig: closed-loop paths (prior / ours / DPCC-C, 2.5 cm) ----------------
cfg = yaml.safe_load(open('config/projection_eval.yaml')); EXP = 'avoiding-d3il'
HS, OB = cfg['halfspace_constraints'][EXP], cfg['obstacle_constraints'][EXP]
LIDX = {'bh': ([2, 3], 5), 'tl': ([0], 3), 'tr': ([1], 4)}
XLIM, YLIM, ROD_R, ENL = (0.25, 0.75), (-0.3, 0.4), 0.01, 0.025
POSTS = [((0.5, -0.1), 0.03), ((0.425, 0.08), 0.025), ((0.575, 0.08), 0.025), ((0.35, 0.26), 0.025), ((0.5, 0.26), 0.025), ((0.65, 0.26), 0.025)]
def draw_scene(ax, l, ENL=ENL):
    hs_idx, ob_idx = LIDX[l]; xs = np.linspace(*XLIM, 200)
    for k in hs_idx:
        (x0, y0), (x1, y1), _ = HS[k]; m = (y1 - y0) / (x1 - x0); c = y0 - m * x0
        ax.fill_between(xs, m * xs + c - ENL * np.sqrt(1 + m * m), YLIM[1], color='0.85', zorder=0, lw=0)
        ax.fill_between(xs, m * xs + c, YLIM[1], color='#b6b6e8', zorder=0, lw=0)
    o = OB[ob_idx]
    ax.add_patch(Circle(o['center'], o['radius'] + ENL, color='0.85', zorder=0, lw=0))
    ax.add_patch(Circle(o['center'], o['radius'], color='#b6b6e8', zorder=0, lw=0))
    for c, r in POSTS: ax.add_patch(Circle(c, r + ROD_R, color='#e8c4c4', zorder=1, lw=0)); ax.add_patch(Circle(c, r, color='#b22222', zorder=2, lw=0))
    ax.axhline(0.35, color='#3cc03c', lw=3, zorder=1); ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
def paths(sw, job):
    P = []
    for f in sorted(glob.glob(f'analysis/stats_{sw}/{job}*/**/*_paths.pkl', recursive=True)):
        if re.search(rf'/{re.escape(job)}(_s\d)?/', f): P += pickle.load(open(f, 'rb'))
    return P
rows = [('Prior', 'Prior'), ('Ours', 'TCA-DPC')]
OURS0P = {l: ('sweep_v19s', f'split_{l}_buf0') for l in ['tl', 'tr', 'bh']}   # gamma = 0 rerun of v16 with paths saved
for SRCP, out, enl in [(OURS0P, 'paper_paths', 0.0), (OURS, 'paper_paths_g25', ENL)]:   # paper: gamma = 0
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.0, 2.45 * len(rows) + 0.35))
    for i, (m, lab) in enumerate(rows):
        for j, l in enumerate(['tl', 'tr', 'bh']):
            ax = axes[i, j]; draw_scene(ax, l, enl); P = paths(*(src(m, l) if m == 'Prior' else SRCP[l]))
            for p in P:     # all episodes in one colour, no success/failure marking (as in DPCC, Fig. 3)
                ax.plot(p['xy'][:, 0], p['xy'][:, 1], color=C['Ours'], lw=0.7, alpha=0.45, zorder=3)
            ax.set_title(LAY[l], pad=2)
            if j == 0: ax.text(-0.04, 0.5, lab, transform=ax.transAxes, rotation=90, ha='right', va='center')
    fig.legend(handles=[Line2D([], [], color=C['Ours'], lw=1.5, label='closed-loop path'),
                        Line2D([], [], color='#b6b6e8', lw=6, label='test-time constraint'),
                        Line2D([], [], color='0.85', lw=6, label=r'tightening $\gamma$')][:3 if enl > 0 else 2],
               loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.055))
    fig.subplots_adjust(wspace=0.04, hspace=0.1, bottom=0.06, top=0.97)
    save(fig, out)

# ---------------- Fig: scenarios x (paths | beta_t | success vs gamma); paths / beta at gamma = 0 ----------------
# drawn at final print size (PMLR \linewidth ~ 6 in), so fonts are true 8-9 pt in the paper
def scen_fig(LAYS, out, figsize, top, bottom):
  with plt.rc_context({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8}):
    fig, axes = plt.subplots(len(LAYS), 3, figsize=figsize, gridspec_kw={'width_ratios': [0.62, 1, 1]}, squeeze=False); n = len(LAYS)
    for i, l in enumerate(LAYS):
        # paths: prior (grey) and ours (blue)
        ax = axes[i, 0]; draw_scene(ax, l, 0.0)
        for s, c, a in [(src('Prior', l), C['Prior'], 0.35), (OURS0P[l], C['Ours'], 0.45)]:
            for p in paths(*s): ax.plot(p['xy'][:, 0], p['xy'][:, 1], color=c, lw=0.6, alpha=a, zorder=3)
        # beta_t, median + IQR, aligned to first failure
        ax = axes[i, 1]; sw, job = OURS0[l]
        T = sum([load(p)[0] for p in sorted(glob.glob(f'analysis/stats_{sw}/{job}_s?'))], [])
        A = np.full((len(T), PRE + POST + 1), np.nan)
        for e, t in enumerate(T):
            t0 = int(np.argmax(t['fail_frac'] > 0)); x = t['beta'][max(t0 - PRE, 0):t0 + POST + 1]
            A[e, PRE - min(t0, PRE):PRE - min(t0, PRE) + len(x)] = x
        keep = (~np.isnan(A)).sum(0) >= NMIN; s_ = np.arange(-PRE, POST + 1); B = np.maximum(A, 1.)
        ax.axvline(0, color='#8a8984', lw=0.5, ls=':')
        ax.fill_between(s_[keep], np.nanpercentile(B, 25, 0)[keep], np.nanpercentile(B, 75, 0)[keep], color=C['Ours'], alpha=0.18, lw=0)
        ax.plot(s_[keep], np.nanmedian(B, 0)[keep], color=C['Ours'], lw=1.2)
        ax.axhline(1e4, color='black', lw=0.5, ls=(0, (4, 2)))
        ax.text(0.97, 1e4 * 0.55, r'$\beta_{\max}$', transform=ax.get_yaxis_transform(), ha='right', va='top')
        ax.set_yscale('log'); ax.set_ylim(0.7, 2e4); ax.set_yticks([1, 1e1, 1e2, 1e3, 1e4])
        ax.set_yticklabels(['$0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$']); ax.minorticks_off(); ax.set_xlim(-PRE, POST)
        ax.set_ylabel(r'$\beta_t$', labelpad=1)
        # success vs tightening gamma
        ax = axes[i, 2]
        for m, dx in [('Ours', -0.03), ('DPCC-C', 0.), ('DPCC-T', 0.03)]:
            y = np.array([ms(*split_src(m, l, b)) for b in B02])
            x = np.array([float(b) for b in B02]) + dx; ok = ~np.isnan(y[:, 0])
            ax.errorbar(x[ok], y[ok, 0], yerr=y[ok, 1], color=C[m], marker=MK[m], ms=3.5, lw=1.0 if ok.all() else 0,
                        capsize=1.5, elinewidth=0.6, mec='white', mew=0.5)
        ax.set_xticks([float(b) for b in B02]); ax.set_ylim(0.3, 1.03); ax.set_ylabel('Success (G&C)', labelpad=1)
        if i < n - 1: axes[i, 1].set_xticklabels([]); axes[i, 2].set_xticklabels([])
    axes[-1, 1].set_xlabel(r'Steps since first failure $t-t_0$'); axes[-1, 2].set_xlabel(r'Tightening $\gamma$ [cm]')
    for j, t in enumerate(['(a) Closed-loop paths', r'(b) Tilt strength $\beta_t$', r'(c) Success vs. $\gamma$']):
        axes[0, j].set_title(t, pad=4)
    fig.subplots_adjust(left=0.06, right=0.99, top=top, bottom=bottom, wspace=0.36, hspace=0.12)
    # paths axes: same height as the row, width from the scene aspect (equal x/y scale), right edge aligned to its slot
    fw, fh = fig.get_size_inches()
    for i, l in enumerate(LAYS):
        p0, p1 = axes[i, 0].get_position(), axes[i, 1].get_position()
        w = p1.height * fh * (XLIM[1] - XLIM[0]) / (YLIM[1] - YLIM[0]) / fw
        axes[i, 0].set_position([p0.x1 - w, p1.y0, w, p1.height])
        axes[i, 0].text(-0.06, 0.5, LAY[l], transform=axes[i, 0].transAxes, rotation=90, ha='right', va='center', fontsize=9)
    fig.legend(handles=[Line2D([], [], color=C['Prior'], lw=1.5, label='Prior'),
                        Line2D([], [], color=C['Ours'], lw=1.2, marker=MK['Ours'], ms=3.5, mec='white', mew=0.5, label='TCA-DPC'),
                        Line2D([], [], color=C['DPCC-C'], lw=1.0, marker=MK['DPCC-C'], ms=3.5, mec='white', mew=0.5, label='DPCC-C'),
                        Line2D([], [], color=C['DPCC-T'], lw=1.0, marker=MK['DPCC-T'], ms=3.5, mec='white', mew=0.5, label='DPCC-T'),
                        Line2D([], [], color='#b6b6e8', lw=6, label='test-time constraint')],
               loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.2)
    save(fig, out)

scen_fig(['tl', 'tr', 'bh'], 'paper_scenarios', (6.0, 5.2), 0.9, 0.08)
for l in ['tl', 'tr', 'bh']: scen_fig([l], f'paper_scenario_{l}', (6.0, 2.2), 0.76, 0.19)
