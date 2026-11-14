import os
import time
import numpy as np
import torch

from diffuser.models.helpers import apply_conditioning


class TiltSampler:
    """
        Test-time constraint handling for a frozen diffusion model by exponentially tilting its output law
        and sampling the tilt exactly in the model's noise space (Project update, Oct 1).

        The frozen sampler is a fixed function U = Psi_c(z) of its noise z ~ N(0, I). With a penalty Phi >= 0 that is
        zero exactly on the feasible set, z ~ pi(z) ∝ exp(-|z|^2 / 2 - beta * Phi(Psi_c(z))) implies
        Psi_c(z) ~ Q_beta ∝ exp(-beta * Phi) P. We run MALA/Langevin on z (backpropagating through the K reverse steps),
        verify every candidate, apply the best verified one, otherwise a fallback, and adapt beta online so that the
        fraction of failing candidates tracks a budget delta. Noise vectors and beta persist across control steps.

        Constraints are taken from a Projector (normalized trajectory space), so the same (tightened) constraints
        as in DPCC are used. Phi is evaluated on the generated state-action trajectory.
    """

    def __init__(self, diffusion, projector, goal_dim=0, n_langevin=5, step_size=0.05, mala=True,
                 tau=0.05, beta_init=5.0, beta_max=50.0, kappa=2.0, delta=0.25, adapt_beta=True,
                 warm_start=True, n_fresh=1, feas_tol=1e-3, dyn_tol=0.05, selection='consistency',
                 fallback='project', grad_clip=1e3, target_accept=None, tau_mode='fixed', tau_min=1e-6,
                 margin=0., hinge_weight=0., prev_max=3, penalty='exp', rho='sq', rho_eps=1e-3,
                 extra_mala=0, fallback_last='project', bounds_tol=None, escalate=0):
        self.diffusion = diffusion
        self.projector = projector
        self.goal_dim = goal_dim
        self.M = n_langevin
        self.h = step_size
        self.mala = mala
        self.tau = tau
        self.beta_init, self.beta_max, self.kappa, self.delta = beta_init, beta_max, kappa, delta
        self.adapt_beta = adapt_beta
        self.warm_start = warm_start
        self.n_fresh = n_fresh
        self.feas_tol = feas_tol
        self.bounds_tol = bounds_tol    # separate check tolerance for box bounds (actuator limits); None = feas_tol
        self._bmask = None
        self.escalate = escalate        # rounds of fresh noise + M MALA steps when nothing passes (before the fallback)
        self.dyn_tol = dyn_tol
        self.selection = selection      # 'consistency' | 'prior' | 'random'
        self.fallback = fallback        # 'prev' (previous plan shifted one step; final algo) | 'project' (DPCC projection of the least-violating candidate) | 'best'
        self.grad_clip = grad_clip
        self.target_accept = target_accept      # e.g. 0.574 (optimal MALA acceptance, Roberts & Rosenthal 1998): adapts the step size across control steps
        self.tau_mode = tau_mode                # 'fixed' | 'median' (tau <- median positive violation S of the initial chains, from the previous control step)
        self.tau_min = tau_min
        self.margin = margin                    # eps: Phi penalizes g + eps > 0 (tightened), verification still at g <= feas_tol
        self.hinge_weight = hinge_weight        # lambda: Phi += lambda * sum [g + eps]_+ (non-saturating gradient far from feasibility)
        self.penalty_mode = penalty             # 'exp': Phi = 1 - exp(-S / tau) (bounded) | 'S': Phi = S (unbounded, no tau)
        self.rho = rho                          # 'sq': rho(a) = [a]_+^2 | 'smooth_hinge': 0 (a<=0), a^2/(2 eps) (0<a<=eps), a - eps/2 (a>eps)
        self.rho_eps = rho_eps
        self.extra_mala = extra_mala            # if no candidate passes after M steps: up to this many extra MALA steps, stop as soon as one passes
        self.fallback_last = fallback_last      # when the previous plan cannot be used: 'project' (DPCC) | 'best' (least-violating candidate, no DPCC)
        self.prev_max = prev_max                # 'prev' fallback: at most this many consecutive reuses of the previous plan, then DPCC projection
        self.reset()

    def reset(self):
        self.z = None
        self.beta = self.beta_init
        self.h_cur = self.h
        self.tau_cur = self.tau
        self.prev_plan = None
        self.n_prev = 0
        self.stats = {'fail_frac': [], 'beta': [], 'fallback': [], 'acc': [], 'h': [], 'tau': [],
                      'beta_next': [], 'n_ok': [], 'vmax_chosen': [], 'vmax_min': [], 'S0_median_pos': [], 'S0_frac_pos': [],
                      'chosen': [], 'sample_time': [], 'fallback_prev': [], 'fallback_ok': [], 'fallback_dpcc': [], 'extra_steps': [], 'rescued': [],
                      'phi_pre': [], 'phi_post': [], 'phi_chosen': [], 'beta_phi_chosen': [], 'prior_term': [], 'vmax_all': []}   # (logging) penalty values per candidate

    # ------------------------------------------------------------------ constraints ------------------------------------------------------------------ #

    def violations(self, x, margin=0., return_hinge=False):
        """
            x: [B, H, transition_dim (incl. goal)] normalized trajectories (differentiable).
            Returns (S, vmax[, H]): S = sum of squared violations of g + margin [B] (used in Phi),
            vmax = largest violation of g, without margin [B] (used to verify), H = sum of [g + margin]_+ [B].
        """
        p = self.projector
        traj = x[:, :, :-self.goal_dim] if self.goal_dim > 0 else x
        flat = traj.reshape(traj.shape[0], -1)
        parts = []
        slack = []                          # per-part extra check tolerance (verification only, not in Phi)
        if p.C.shape[0] > 0:
            gC = flat @ p.C.T - p.d
            if self.bounds_tol is not None and self.bounds_tol > self.feas_tol:
                if self._bmask is None:     # box-bound rows have one nonzero entry; halfspace rows have two
                    self._bmask = (p.C != 0).sum(dim=1) == 1
                parts += [gC[:, self._bmask], gC[:, ~self._bmask]]; slack += [self.bounds_tol - self.feas_tol, 0.]
            else:
                parts.append(gC); slack.append(0.)
        if p.A.shape[0] > 0:
            b = p.b.unsqueeze(0).repeat(flat.shape[0], 1)
            if p.skip_initial_state:        # rows fixing the initial state, as in Projector.project
                s0 = flat[:, :p.transition_dim].detach()
                for counter, c in enumerate([c for c in p.dynamic_constraints.constraint_list if c[0] == 'deriv']):
                    b[:, counter * p.horizon] = s0[:, int(c[1][0])]
            parts.append((flat @ p.A.T - b).abs() - self.dyn_tol); slack.append(0.)
        for P, q, v in zip(p.obstacle_constraints.P_list, p.obstacle_constraints.q_list, p.obstacle_constraints.v_list):
            P = torch.as_tensor(P, dtype=flat.dtype, device=flat.device)
            q = torch.as_tensor(q, dtype=flat.dtype, device=flat.device)
            s = traj[:, 1:]                                              # obstacle constraints skip t = 0
            val = torch.einsum('bhi,ij,bhj->bh', s, P, s) + s @ q
            parts.append(val - float(v)); slack.append(0.)
        parts, slack = [g for g in parts if g.shape[1] > 0], [e for g, e in zip(parts, slack) if g.shape[1] > 0]
        if len(parts) == 0:
            zero = torch.zeros(x.shape[0], device=x.device)
            return (zero, zero, zero) if return_hinge else (zero, zero)
        S = sum(self._rho(g + margin).sum(dim=1) for g in parts)
        vmax = torch.stack([torch.relu(g - e).max(dim=1).values for g, e in zip(parts, slack)], dim=1).max(dim=1).values
        if return_hinge:
            return S, vmax, sum(torch.relu(g + margin).sum(dim=1) for g in parts)
        return S, vmax

    def group_violations(self, x):
        """(logging) largest violation per constraint group for one plan x [1, H, D]: [halfspace/bounds, dynamics, obstacle_1, ...]"""
        p = self.projector
        traj = x[:, :, :-self.goal_dim] if self.goal_dim > 0 else x
        flat = traj.reshape(traj.shape[0], -1)
        out = [float(torch.relu(flat @ p.C.T - p.d).max()) if p.C.shape[0] > 0 else 0.]
        if p.A.shape[0] > 0:
            b = p.b.unsqueeze(0).repeat(flat.shape[0], 1)
            if p.skip_initial_state:
                s0 = flat[:, :p.transition_dim]
                for counter, c in enumerate([c for c in p.dynamic_constraints.constraint_list if c[0] == 'deriv']):
                    b[:, counter * p.horizon] = s0[:, int(c[1][0])]
            out.append(float(torch.relu((flat @ p.A.T - b).abs() - self.dyn_tol).max()))
        else:
            out.append(0.)
        for P, q, v in zip(p.obstacle_constraints.P_list, p.obstacle_constraints.q_list, p.obstacle_constraints.v_list):
            P = torch.as_tensor(P, dtype=flat.dtype, device=flat.device); q = torch.as_tensor(q, dtype=flat.dtype, device=flat.device)
            s = traj[:, 1:]
            out.append(float(torch.relu(torch.einsum('bhi,ij,bhj->bh', s, P, s) + s @ q - float(v)).max()))
        return np.array(out)

    def _rho(self, a):
        if self.rho == 'smooth_hinge':
            e = self.rho_eps
            a = torch.relu(a)
            return torch.where(a <= e, a ** 2 / (2 * e), a - e / 2)
        return torch.relu(a) ** 2

    def penalty(self, x, tau=None):
        S, vmax, H = self.violations(x, margin=self.margin, return_hinge=True)
        self._last_S = S.detach()
        phi = S if self.penalty_mode == 'S' else 1. - torch.exp(-S / (self.tau if tau is None else tau))
        if self.hinge_weight > 0:
            phi = phi + self.hinge_weight * H
        return phi, vmax

    # ------------------------------------------------------------------ sampling ------------------------------------------------------------------ #

    def _potential(self, z, cond, returns, beta):
        x = self.diffusion.sample_from_noise(z, cond, returns)
        phi, vmax = self.penalty(x, self.tau_cur)
        V = 0.5 * (z ** 2).flatten(1).sum(dim=1) + beta * phi
        return V, x, vmax

    def _value_and_grad(self, z, cond, returns, beta):
        z = z.detach().requires_grad_(True)
        with torch.enable_grad():
            V, x, vmax = self._potential(z, cond, returns, beta)
            g, = torch.autograd.grad(V.sum(), z)
        g = torch.nan_to_num(g)
        nrm = g.flatten(1).norm(dim=1).clamp(min=1e-12)
        g = g * (self.grad_clip / nrm).clamp(max=1.).view(-1, 1, 1, 1)
        return V.detach(), g, x.detach(), vmax.detach()

    def _init_noise(self, B, shape, device):
        if self.z is None or not self.warm_start or self.z.shape[0] != B:
            return torch.randn(B, *shape, device=device)
        z = self.z.clone()
        fresh = torch.randperm(B, device=device)[:self.n_fresh]
        z[fresh] = torch.randn(len(fresh), *shape, device=device)
        return z

    def sample(self, cond, returns=None, batch_size=4):
        """
            Run M MALA steps on each of B noise vectors and select a verified candidate.
            Returns (x [B, H, T] normalized, info dict). x[info['chosen']] is the trajectory to execute.
        """
        t_start = time.time()
        d = self.diffusion
        device = d.betas.device
        shape = (d.n_timesteps + 1, d.horizon, d.transition_dim)
        z = self._init_noise(batch_size, shape, device)
        beta = self.beta
        V, g, x, vmax = self._value_and_grad(z, cond, returns, beta)
        S0 = self._last_S.clone()
        n_acc, n_prop = 0, 0

        def mala_step(z, g, x, V, vmax):
            zeta = torch.randn_like(z)
            z_prop = z - self.h_cur * g + (2 * self.h_cur) ** 0.5 * zeta
            V_p, g_p, x_p, vmax_p = self._value_and_grad(z_prop, cond, returns, beta)
            if self.mala:
                # log q(z|z') - log q(z'|z) with q(z'|z) = N(z - h grad V(z), 2h I)
                fwd = (z_prop - z + self.h_cur * g).flatten(1).pow(2).sum(1)
                bwd = (z - z_prop + self.h_cur * g_p).flatten(1).pow(2).sum(1)
                log_a = (V - V_p) + (fwd - bwd) / (4 * self.h_cur)
                acc = torch.log(torch.rand_like(log_a)) < log_a
            else:
                acc = torch.ones(batch_size, dtype=torch.bool, device=device)
            z = torch.where(acc.view(-1, 1, 1, 1), z_prop, z)
            g = torch.where(acc.view(-1, 1, 1, 1), g_p, g)
            x = torch.where(acc.view(-1, 1, 1), x_p, x)
            V = torch.where(acc, V_p, V)
            vmax = torch.where(acc, vmax_p, vmax)
            return z, g, x, V, vmax, int(acc.sum())

        for _ in range(self.M):
            z, g, x, V, vmax, a = mala_step(z, g, x, V, vmax)
            n_acc += a; n_prop += batch_size

        # ---- verify (f_t measures the M-step sampler), extra MALA steps if nothing passes, select, fallback
        ok = vmax <= self.feas_tol
        f_t = 1. - ok.float().mean().item()
        n_extra = 0
        if not ok.any() and self.extra_mala > 0:
            for n_extra in range(1, self.extra_mala + 1):
                z, g, x, V, vmax, a = mala_step(z, g, x, V, vmax)
                n_acc += a; n_prop += batch_size
                ok = vmax <= self.feas_tol
                if ok.any(): break
        n_esc = 0
        if not ok.any() and self.escalate > 0:     # search harder only when stuck: fresh noises, same tilt
            for n_esc in range(1, self.escalate + 1):
                z = torch.randn(batch_size, *shape, device=device)
                V, g, x, vmax = self._value_and_grad(z, cond, returns, beta)
                for _ in range(self.M):
                    z, g, x, V, vmax, a = mala_step(z, g, x, V, vmax)
                    n_acc += a; n_prop += batch_size
                ok = vmax <= self.feas_tol
                if ok.any(): break
        self.stats.setdefault('esc_rounds', []).append(n_esc)
        rescued = (n_extra > 0 or n_esc > 0) and bool(ok.any())
        if os.environ.get('DPCC_DEBUG_VIOL', '0') == '1':      # logging only: best candidate's plan and per-group violation
            ib = int(torch.argmin(vmax))
            with torch.no_grad():
                self.stats.setdefault('dbg_groups', []).append(self.group_violations(x[ib:ib + 1].detach()))
            self.stats.setdefault('dbg_best_x', []).append(x[ib].detach().cpu().numpy())
        fallback_used, fb = False, None
        if ok.any():
            chosen = self._select(x, z, ok)
            self.n_prev = 0
        else:
            fallback_used = True
            chosen = int(torch.argmin(vmax))
            fb = self.fallback
            if fb == 'prev' and (self.prev_plan is None or self.n_prev >= self.prev_max):
                fb = self.fallback_last         # no previous plan yet, or it has been reused prev_max times in a row
            self.n_prev = self.n_prev + 1 if fb == 'prev' else 0
            if fb == 'prev':
                # keep the previous plan: shift by one step, repeat the last step, re-impose the current state
                shifted = torch.cat([self.prev_plan[1:], self.prev_plan[-1:]], dim=0).unsqueeze(0)
                x = x.clone()
                x[chosen:chosen + 1] = apply_conditioning(shifted.to(x.dtype), {k: v[chosen:chosen + 1] for k, v in cond.items() if not isinstance(k, str)},
                                                          d.action_dim, goal_dim=d.goal_dim)
            elif fb == 'project':
                traj = x[chosen:chosen + 1]
                body = traj[:, :, :-self.goal_dim] if self.goal_dim > 0 else traj
                proj, _ = self.projector.project(body, None)
                x = x.clone()
                if self.goal_dim > 0:
                    x[chosen:chosen + 1, :, :-self.goal_dim] = proj.to(x.dtype)
                else:
                    x[chosen:chosen + 1] = proj.to(x.dtype)
                x[chosen:chosen + 1] = apply_conditioning(x[chosen:chosen + 1].clone(), {k: v[chosen:chosen + 1] for k, v in cond.items() if not isinstance(k, str)},
                                                          d.action_dim, goal_dim=d.goal_dim)

        # ---- heuristics: step-size adaptation toward the optimal MALA acceptance, median heuristic for tau
        acc_rate = n_acc / max(n_prop, 1)
        if self.target_accept is not None and self.mala:
            self.h_cur = float(min(1.0, max(1e-3, self.h_cur * np.exp(0.5 * (acc_rate - self.target_accept)))))
        if self.tau_mode == 'median':
            pos = S0[S0 > 0]
            if pos.numel() > 0:
                self.tau_cur = float(max(self.tau_min, pos.median().item()))

        # ---- online tilt (thermostat on the failure fraction) and persistence
        if self.adapt_beta:
            self.beta = float(min(self.beta_max, max(0., self.beta + self.kappa * (f_t - self.delta))))
        self.z = z.detach()
        self.prev_plan = x[chosen].detach().clone()
        self.stats['fail_frac'].append(f_t); self.stats['beta'].append(beta); self.stats['fallback'].append(float(fallback_used))
        self.stats['acc'].append(acc_rate); self.stats['h'].append(self.h_cur); self.stats['tau'].append(self.tau_cur)
        pos0 = S0[S0 > 0]
        self.stats['beta_next'].append(self.beta); self.stats['n_ok'].append(int(ok.sum()))
        self.stats['vmax_chosen'].append(float(vmax[chosen])); self.stats['vmax_min'].append(float(vmax.min()))
        self.stats['S0_median_pos'].append(float(pos0.median()) if pos0.numel() > 0 else 0.)
        self.stats['S0_frac_pos'].append(float((S0 > 0).float().mean())); self.stats['chosen'].append(int(chosen))
        self.stats['sample_time'].append(time.time() - t_start)
        self.stats['fallback_prev'].append(float(fallback_used and fb == 'prev'))
        self.stats['fallback_dpcc'].append(float(fallback_used and fb == 'project'))
        self.stats['extra_steps'].append(n_extra); self.stats['rescued'].append(float(rescued))
        with torch.no_grad():       # (logging only) Phi before / after MALA per candidate, tilt term vs prior term
            S_post, _ = self.violations(x, margin=self.margin)
            to_phi = lambda S: S if self.penalty_mode == 'S' else 1. - torch.exp(-S / self.tau_cur)
            phi_post = to_phi(S_post)
            self.stats['phi_pre'].append(to_phi(S0).cpu().numpy()); self.stats['phi_post'].append(phi_post.cpu().numpy())
            self.stats['phi_chosen'].append(float(phi_post[chosen])); self.stats['beta_phi_chosen'].append(float(beta * phi_post[chosen]))
            self.stats['prior_term'].append((0.5 * (z ** 2).flatten(1).sum(dim=1)).cpu().numpy()); self.stats['vmax_all'].append(vmax.cpu().numpy())
        if fallback_used:           # re-verify the fallback plan against the same constraints
            with torch.no_grad():
                _, v_fb = self.violations(x[chosen:chosen + 1])
            self.stats['fallback_ok'].append(float(v_fb.item() <= self.feas_tol))
        else:
            self.stats['fallback_ok'].append(float('nan'))
        info = {'vmax': vmax.cpu().numpy(), 'chosen': chosen, 'fail_frac': f_t, 'beta': beta, 'fallback': fallback_used, 'ok': ok.cpu().numpy()}
        return x, info

    def _select(self, x, z, ok):
        idx = torch.nonzero(ok).flatten()
        if self.selection == 'consistency' and self.prev_plan is not None:
            # distance to the previous plan shifted by one step (states only)
            ad = self.diffusion.action_dim
            prev = self.prev_plan[1:, ad:]
            cur = x[idx][:, :-1, ad:]
            n = min(prev.shape[-1], cur.shape[-1])
            cost = (cur[..., :n] - prev[..., :n]).flatten(1).norm(dim=1)
        elif self.selection == 'prior':
            cost = (z[idx] ** 2).flatten(1).sum(1)
        else:
            return int(idx[0])
        return int(idx[torch.argmin(cost)])
