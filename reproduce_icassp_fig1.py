#!/usr/bin/env python3
import argparse, csv, json, math, os, platform, sys, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 8.5
plt.rcParams['axes.titlesize'] = 9.8
plt.rcParams['axes.labelsize'] = 8.7
plt.rcParams['xtick.labelsize'] = 8.0
plt.rcParams['ytick.labelsize'] = 8.0
plt.rcParams['legend.fontsize'] = 8.0
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
from matplotlib.ticker import FixedLocator, FuncFormatter
from numba import njit

BASE_SEED = 2026091500
NOISE_SCALE = 1e-4
DIMS = [2, 3, 5, 10, 20]
PERT_REPS = 12
T_COMPARE = 350_000
T_PERT = 200_000
T_INIT = 200_000
K = 3
P = 0.6
RHO = 0.2


def stable_psi(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    ep = np.exp(-x[pos])
    out[pos] = ep / (1.0 + ep)
    en = np.exp(x[~pos])
    out[~pos] = 1.0 / (1.0 + en)
    return out


def center_data(d):
    if d < 2:
        raise ValueError('d must be >= 2')
    z = np.zeros((5, d), dtype=np.float64)
    z[0, :2] = [1.0, 0.0]                 # r
    z[1, :2] = [0.0, 1.0]                 # z2
    z[2, :2] = [3.0/5.0, -4.0/5.0]        # z3 = s
    u = np.array([8.0, 1.0]) / math.sqrt(65.0)
    z[3, :2] = u
    z[4, :2] = u
    return z


def theorem_certificate(z, tol=1e-12):
    # Intended main-theorem partition: r=0, C={1,2}, J={3,4}.
    r = 0; C = [1, 2]; J = [3, 4]
    S = z[[r] + C].sum(axis=0)
    a = float(z[r] @ S)
    gamma = []
    xi = []
    zeta = []
    for j in J:
        gamma.append(float(z[j] @ S + sum(min(0.0, float(z[j] @ z[l])) for l in J)))
        xi.append(float((z[j]-z[r]) @ S + sum(min(0.0, float((z[j]-z[r]) @ z[l])) for l in J)))
    for i in C:
        zeta.append(float((z[r]-z[i]) @ S + sum(min(0.0, float((z[r]-z[i]) @ z[l])) for l in J)))
    projections = z @ S
    ok = (
        a > tol and min(gamma) > tol and min(xi) > tol and min(zeta) >= -tol
        and np.min(projections) > tol
    )
    return {
        'ok': bool(ok), 'a': a, 'gamma': gamma, 'xi': xi, 'zeta': zeta,
        'S': S.tolist(), 'min_projection_on_S': float(np.min(projections))
    }


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def angle_deg(v, target):
    v = unit(v); target = unit(target)
    return float(np.degrees(np.arccos(np.clip(v @ target, -1.0, 1.0))))


def threshold(g, rule, rho=RHO):
    if rule == 'gd':
        return math.inf
    if rule == 'fixed':
        return float(rho)
    gs = np.sort(g)
    if rule == 'type1':
        return float(gs[K-1])
    if rule == 'type7':
        # n=5, p=.6 => h=1+(n-1)p=3.4 => .6*x_(3)+.4*x_(4)
        return float(0.6*gs[2] + 0.4*gs[3])
    raise ValueError(rule)


@njit(cache=True)
def _psi_scalar(x):
    if x >= 0.0:
        e=math.exp(-x)
        return e/(1.0+e)
    e=math.exp(x)
    return 1.0/(1.0+e)

@njit(cache=True)
def _run_terminal(z,T,eta,rule_code,rho,w0):
    n,d=z.shape
    w=w0.copy()
    tau=0.0
    for _ in range(T):
        g=np.empty(n)
        for i in range(n):
            m=0.0
            for j in range(d): m += z[i,j]*w[j]
            g[i]=_psi_scalar(m)
        if rule_code==0: # gd
            tau=1e300
        elif rule_code==1: # fixed
            tau=rho
        else:
            gs=np.sort(g.copy())
            if rule_code==2: tau=gs[2]
            else: tau=0.6*gs[2]+0.4*gs[3]
        for j in range(d):
            u=0.0
            for i in range(n):
                c=g[i] if g[i] < tau else tau
                u += c*z[i,j]
            w[j] += eta/n*u
    return w,tau

@njit(cache=True)
def _run_batch_exact(zbatch,T,etas,w0batch):
    B,n,d=zbatch.shape
    out=np.empty((B,d)); qs=np.empty(B)
    for b in range(B):
        w,q=_run_terminal(zbatch[b],T,etas[b],2,0.0,w0batch[b])
        out[b]=w; qs[b]=q
    return out,qs

def run(z, T, eta, rule='type1', w0=None, rho=RHO, checkpoints=None):
    z = np.asarray(z, dtype=np.float64)
    n, d = z.shape
    w = np.zeros(d, dtype=np.float64) if w0 is None else np.asarray(w0, dtype=np.float64).copy()
    if checkpoints is None:
        checkpoints = []
    cp = set(int(x) for x in checkpoints)
    rec = []
    tau_last = None
    for t in range(1, int(T)+1):
        g = stable_psi(z @ w)  # all examples unit norm in reported experiments
        tau = threshold(g, rule, rho)
        coeff = g if not np.isfinite(tau) else np.minimum(g, tau)
        w += (float(eta)/n) * (coeff[:, None] * z).sum(axis=0)
        if not np.all(np.isfinite(w)):
            raise FloatingPointError(f'nonfinite iterate at t={t}')
        tau_last = tau
        if t in cp:
            rec.append((t, w.copy(), float(tau), g.copy()))
    return w, tau_last, rec



def run_batch_exact(zbatch, T, etas, w0batch=None):
    zbatch=np.asarray(zbatch,dtype=np.float64)
    B,n,d=zbatch.shape
    etas=np.broadcast_to(np.asarray(etas,dtype=np.float64).reshape(-1), (B,)).copy()
    if w0batch is None:
        w0batch=np.zeros((B,d),dtype=np.float64)
    else:
        w0batch=np.asarray(w0batch,dtype=np.float64)
    return _run_batch_exact(zbatch,int(T),etas,w0batch)

def log_checkpoints(T, m=220):
    vals = np.unique(np.rint(np.logspace(0, math.log10(T), m)).astype(int))
    vals = vals[(vals >= 1) & (vals <= T)]
    if vals[-1] != T:
        vals = np.append(vals, T)
    return vals


def comparison(outdir, quick=False):
    T = 20_000 if quick else T_COMPARE
    z = center_data(2)
    mm = unit(np.array([3.0, 1.0]))
    active = unit(np.array([8.0, 1.0]))
    cps = log_checkpoints(T)
    rows = []
    trajectories = []
    for rule in ['gd', 'fixed', 'type1', 'type7']:
        w, tau, rec = run(z, T, 1.0, rule=rule, rho=RHO, checkpoints=cps)
        rows.append({
            'rule': rule,
            'T': T,
            'eta': 1.0,
            'p': P,
            'rho': RHO if rule == 'fixed' else '',
            'angle_to_maxmargin_deg': angle_deg(w, mm),
            'angle_to_active_deg': angle_deg(w, active),
            'terminal_tau': tau if np.isfinite(tau) else '',
            'T_times_tau': T*tau if np.isfinite(tau) else '',
        })
        for t, wt, taut, g in rec:
            trajectories.append({
                'rule': rule, 'iteration': t,
                'angle_to_maxmargin_deg': angle_deg(wt, mm),
                'angle_to_active_deg': angle_deg(wt, active),
                'tau': taut if np.isfinite(taut) else '',
                't_times_tau': t*taut if np.isfinite(taut) else '',
            })
    return rows, trajectories


def perturbation_sweep(outdir, quick=False):
    T = 20_000 if quick else T_PERT
    reps = 2 if quick else PERT_REPS
    rows = []
    for d in DIMS:
        zs=[]; metas=[]; Ss=[]
        for rep in range(reps):
            seed = BASE_SEED + 100*d + rep
            rng = np.random.default_rng(seed)
            z = center_data(d)
            z += NOISE_SCALE * rng.normal(size=z.shape)
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            cert = theorem_certificate(z)
            if not cert['ok']:
                raise RuntimeError(f'certificate failed for d={d}, rep={rep}, seed={seed}: {cert}')
            zs.append(z); Ss.append(np.asarray(cert['S'])); metas.append((rep,seed,cert))
        wb,qb=run_batch_exact(np.stack(zs),T,np.ones(reps))
        for b,(rep,seed,cert) in enumerate(metas):
            rows.append({
                'd': d, 'rep': rep, 'seed': seed, 'noise_scale': NOISE_SCALE,
                'T': T, 'eta': 1.0, 'certificate_ok': True,
                'angle_to_predicted_deg': angle_deg(wb[b], Ss[b]),
                'a': cert['a'], 'min_gamma': min(cert['gamma']),
                'min_xi': min(cert['xi']), 'min_zeta': min(cert['zeta']),
                'min_projection_on_S': cert['min_projection_on_S'],
            })
    return rows

def step_size_sweep(outdir, quick=False):
    T = 20_000 if quick else T_COMPARE
    d = 5; rep = 0; seed = BASE_SEED + 100*d + rep
    rng = np.random.default_rng(seed)
    z = center_data(d) + NOISE_SCALE*rng.normal(size=(5,d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    cert = theorem_certificate(z)
    if not cert['ok']:
        raise RuntimeError(f'd=5 certificate failed: {cert}')
    S = np.asarray(cert['S'])
    etas=np.array([0.1,1.0,5.0])
    zb=np.repeat(z[None,:,:],3,axis=0)
    wb,qb=run_batch_exact(zb,T,etas)
    rows=[]
    for b,eta in enumerate(etas):
        rows.append({'d':d,'rep':rep,'seed':seed,'T':T,'eta':float(eta),
                     'angle_to_predicted_deg':angle_deg(wb[b],S),'terminal_tau':float(qb[b])})
    return rows

def initialization_sweep(outdir, quick=False):
    # Fully specified finite-horizon stress test inside the exact theorem cone.
    # Direction is uniform in the cone's angular interval; radius is U[0,1].
    T = 20_000 if quick else T_INIT
    N = 5 if quick else 20
    seed = BASE_SEED + 999
    rng = np.random.default_rng(seed)
    z = center_data(2)
    S = np.array([8.0/5.0, 1.0/5.0])
    ux, uy = 8.0/math.sqrt(65.0), 1.0/math.sqrt(65.0)
    theta_lo = math.atan((1.0-ux)/uy)
    theta_hi = math.pi/4.0
    w0s=[]; meta=[]
    for rep in range(N):
        theta = rng.uniform(theta_lo, theta_hi)
        radius = rng.uniform(0.0, 1.0)
        w0 = radius*np.array([math.cos(theta), math.sin(theta)])
        r=0; J=[3,4]; C=[1,2]
        assert all((z[j]-z[r])@w0 >= -1e-12 for j in J)
        assert all((z[r]-z[i])@w0 >= -1e-12 for i in C)
        w0s.append(w0); meta.append((rep,theta,radius,w0))
    zb=np.repeat(z[None,:,:],N,axis=0)
    wb,qb=run_batch_exact(zb,T,np.ones(N),np.stack(w0s))
    rows=[]
    for b,(rep,theta,radius,w0) in enumerate(meta):
        rows.append({'rep':rep,'seed':seed,'T':T,'eta':1.0,'theta_rad':theta,
                     'radius':radius,'w0_x':w0[0],'w0_y':w0[1],
                     'angle_to_predicted_deg':angle_deg(wb[b],S)})
    return rows

def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text('')
        return
    keys = list(rows[0].keys())
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def make_figure(outdir, comp_rows, trajectories, pert_rows):
    outdir = Path(outdir)
    fig, axs = plt.subplots(1, 3, figsize=(7.9, 2.45), constrained_layout=True)

    # Common styling
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.tick_params(width=0.8, length=3.5)

    # (a) geometry
    ax = axs[0]
    z = center_data(2)
    asdir = unit(np.array([8.0, 1.0]))
    mm = unit(np.array([3.0, 1.0]))

    label_specs = {
        'r': (z[0], (0.03, 0.02)),
        'z2': (z[1], (0.02, 0.05)),
        'z3': (z[2], (0.02, -0.02)),
        'u': (z[3], (0.03, 0.02)),
    }
    for lab, (v, delta) in label_specs.items():
        ax.annotate('', xy=v, xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle='-|>', lw=1.15, color='black',
                                    shrinkA=0, shrinkB=0, mutation_scale=8))
        ax.text(v[0] + delta[0], v[1] + delta[1], lab, fontsize=8.4)

    ax.plot([0, asdir[0]], [0, asdir[1]], linestyle='--', color='C0', label='Type 1 limit')
    ax.plot([0, mm[0]], [0, mm[1]], linestyle=':', color='C1', label='Type 7 / max-margin')
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.9, 1.08)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('(a) Geometry', pad=3.5)
    ax.set_xlabel('first coordinate')
    ax.set_ylabel('second coordinate')
    ax.legend(loc='upper left', frameon=False, handlelength=2.6, borderaxespad=0.2)

    # (b) trajectories
    ax = axs[1]
    style = {
        'type1': dict(linestyle='--', color='C0', label='Type 1, p=0.6'),
        'type7': dict(linestyle='-', color='C1', label='Type 7, p=0.6'),
        'gd': dict(linestyle=':', color='C2', label='GD'),
        'fixed': dict(linestyle='-.', color='C3', label='fixed cap'),
    }
    for rule in ['type1', 'type7', 'gd', 'fixed']:
        rr = [r for r in trajectories if r['rule'] == rule]
        x = [r['iteration'] for r in rr]
        y = [r['angle_to_maxmargin_deg'] for r in rr]
        ax.plot(x, y, **style[rule])
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator([1, 100, 10000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: {1: '1', 100: '100', 10000: '10k'}.get(int(x), '')))
    ax.set_ylim(-0.6, 12.2)
    ax.grid(axis='y', color='0.88', linewidth=0.6)
    ax.set_title('(b) Same nominal quantile', pad=3.5)
    ax.set_xlabel('iteration (log scale)')
    ax.set_ylabel('angle to max-margin (deg.)')
    ax.legend(loc='upper right', frameon=False, handlelength=2.8, borderaxespad=0.2)

    # (c) perturbations
    ax = axs[2]
    dims = []
    meds = []
    for d in DIMS:
        vals = np.array([r['angle_to_predicted_deg'] for r in pert_rows if r['d'] == d])
        med = np.median(vals)
        lo = med - np.min(vals)
        hi = np.max(vals) - med
        dims.append(d)
        meds.append(med)
        ax.errorbar(d, med, yerr=np.array([[lo], [hi]]), fmt='o', capsize=4, markersize=6)
    ax.set_xticks(DIMS)
    ax.set_ylim(-0.001, 0.0275)
    ax.grid(axis='y', color='0.88', linewidth=0.6)
    ax.set_title('(c) 12 perturbations / dimension', pad=3.5)
    ax.set_xlabel('ambient dimension d')
    ax.set_ylabel('angle to predicted direction (deg.)')

    fig.savefig(outdir / 'main_figure.png', dpi=320, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(outdir / 'main_figure.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true',help='short smoke run')
    ap.add_argument('--output',default='results',help='output directory')
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    comp,traj=comparison(out,args.quick)
    pert=perturbation_sweep(out,args.quick)
    steps=step_size_sweep(out,args.quick)
    inits=initialization_sweep(out,args.quick)
    write_csv(out/'comparison_summary.csv',comp)
    write_csv(out/'comparison_trajectories.csv',traj)
    write_csv(out/'perturbation_sweep.csv',pert)
    write_csv(out/'step_size_sweep.csv',steps)
    write_csv(out/'initialization_sweep.csv',inits)
    make_figure(out,comp,traj,pert)
    env={
        'python':sys.version,'numpy':np.__version__,'matplotlib':plt.matplotlib.__version__,
        'platform':platform.platform(),'float_dtype':'float64','base_seed':BASE_SEED,
        'noise_scale':NOISE_SCALE,'elapsed_seconds':time.time()-t0,'quick':args.quick,
    }
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    summary={
        'comparison':comp,
        'perturbation_by_dimension':{},
        'step_size':steps,
        'initialization':{
            'protocol':'20 directions uniform in certified cone angle; radii U[0,1]; seed BASE_SEED+999',
            'median_angle_deg':float(np.median([r['angle_to_predicted_deg'] for r in inits])),
            'max_angle_deg':float(np.max([r['angle_to_predicted_deg'] for r in inits])),
        }
    }
    for d in DIMS:
        vals=np.array([r['angle_to_predicted_deg'] for r in pert if r['d']==d])
        summary['perturbation_by_dimension'][str(d)]={
            'n':int(len(vals)),'min':float(vals.min()),'median':float(np.median(vals)),'max':float(vals.max())
        }
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print(f'\nWrote results to: {out.resolve()}')

if __name__=='__main__':
    main()
