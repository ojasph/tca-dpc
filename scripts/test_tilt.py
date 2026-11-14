"""Smoke test / sanity checks for the noise-space tilt sampler (no closed-loop rollout).
Usage: python scripts/test_tilt.py   (env DPCC_LOGBASE etc. as for eval.py)"""
import os, time, yaml
import numpy as np
import torch
import diffuser.utils as utils
from diffuser.sampling import Policy, Projector, TiltSampler
from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

config = yaml.safe_load(open(os.environ.get('DPCC_EVAL_CONFIG', 'config/projection_eval.yaml')))
exp = 'avoiding-d3il'
enlarge = config['enlarge_constraints']['avoiding']

class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp

args = Parser().parse_args(experiment='plan', seed=0)
exp_ = utils.load_diffusion(args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed), epoch=args.diffusion_epoch, device=args.device)
diffusion, dataset = exp_.diffusion, exp_.dataset
action_dim = diffusion.action_dim
obs_idx = {k: v + action_dim for k, v in config['observation_indices']['avoiding'].items()}
act_obs_idx = {**config['action_indices']['avoiding'], **obs_idx}
tdim = diffusion.transition_dim - diffusion.goal_dim

hs = [config['halfspace_constraints'][exp][2], config['halfspace_constraints'][exp][3]]
obst = [config['obstacle_constraints'][exp][5]]
cl = [('ineq', utils.formulate_halfspace_constraints(c, enlarge, tdim, act_obs_idx)) for c in hs]
lb, ub = utils.formulate_bounds_constraints(config['constraint_types'], config['bounds'][exp], tdim, act_obs_idx)
cl += [['lb', lb], ['ub', ub]]
cl += [[o['type'], [act_obs_idx[o['dimensions'][0]], act_obs_idx[o['dimensions'][1]]], o['center'], o['radius'] + enlarge] for o in obst]
cl += utils.formulate_dynamics_constraints(exp, act_obs_idx, action_dim)
projector = Projector(horizon=args.horizon, transition_dim=tdim, action_dim=action_dim, goal_dim=diffusion.goal_dim, constraint_list=cl,
                      normalizer=dataset.normalizer, variant='states_actions', dt=config['dt']['avoiding'], device=args.device, solver='scipy')

env = ObstacleAvoidanceEnv(); env.start()
obs = env.reset(); a0 = env.robot_state()[:2]
obs = np.concatenate((a0[:2], obs))

# 1) differentiability + determinism of Psi(z)
tilt = TiltSampler(diffusion, projector, goal_dim=diffusion.goal_dim, **config.get('tilt', {}))
policy = Policy(model=diffusion, normalizer=dataset.normalizer, preprocess_fns=args.preprocess_fns, test_ret=args.test_ret, projector=projector, tilt=tilt)
cond = policy._format_conditions({0: policy.preprocess_fn(obs)}, 4)
z = torch.randn(4, diffusion.n_timesteps + 1, diffusion.horizon, diffusion.transition_dim, device=args.device)
x1 = diffusion.sample_from_noise(z, cond); x2 = diffusion.sample_from_noise(z, cond)
print('deterministic:', torch.allclose(x1, x2))
zg = z.clone().requires_grad_(True)
phi, vmax = tilt.penalty(diffusion.sample_from_noise(zg, cond))
phi.sum().backward()
print('grad norm wrt z:', zg.grad.norm().item(), ' phi:', phi.detach().cpu().numpy(), ' vmax:', vmax.detach().cpu().numpy())

# 2) beta = 0 vs beta > 0: violation statistics of candidates
for beta in [0., 5., 20., 50.]:
    t = TiltSampler(diffusion, projector, goal_dim=diffusion.goal_dim, adapt_beta=False, beta_init=beta, **{k: v for k, v in config.get('tilt', {}).items() if k not in ('beta_init', 'adapt_beta')})
    fails, tm = [], time.time()
    for _ in range(5):
        t.reset(); t.beta = beta
        _, info = t.sample(cond, batch_size=4)
        fails.append(info['fail_frac'])
    print(f'beta={beta:5.1f}: failing fraction {np.mean(fails):.2f}, MALA acc {np.mean(t.stats["acc"]):.2f}, {1000*(time.time()-tm)/5:.0f} ms / control step')

# 3) one policy call
action, samples = policy(conditions={0: obs}, batch_size=4, horizon=args.horizon)
print('action', action, 'info', {k: v for k, v in policy.last_tilt_info.items() if k != 'ok'})
