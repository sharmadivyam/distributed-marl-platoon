"""
Distributed training on MPI: every car learns its own IA2C policy inside its own process.

    mpirun --oversubscribe -np 9 python mpi/train_mpi.py --seed 12 --mode independent --out mpi/train/indep_s12
    mpirun --oversubscribe -np 9 python mpi/train_mpi.py --seed 12 --mode consensus  --out mpi/train/cons_s12

Rank 0 is the road. It runs the same episodes as the original trainer (utils.Trainer): a training
episode, then a deterministic test episode, and it keeps the best version by the rolling mean of
the last 5 test rewards. Ranks 1..8 are the cars. Each car has its own network, its own optimizer
and its own experience buffer, and learns only from what it saw itself plus what its neighbours
sent it. Nothing about learning is centralised.

  --mode independent   each car learns alone
  --mode consensus     after every update, neighbours exchange their LSTM weights and average them
                       (the consensus idea of Zhang et al. 2018 and of IA2C-CU in this code base)
  --clip global        (default) gradient clipping uses the size of ALL cars' gradients together, as in
                       the original code: one MPI Allgather of 10 numbers per car per update
  --clip local         each car clips its own gradient (cheaper, but each car can take bigger steps)

At the end the road writes a normal checkpoint (run_dir/model/best/), so the result can be
evaluated with the original code:  python main.py --base-dir <out> evaluate --checkpoint best
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
from mpi4py import MPI

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'mpi'))
import platoon_mpi as pm  # noqa: E402  (V2V layer, local_state, RoadEnv, tags)
from envs.cacc_env import CACCEnv  # noqa: E402

TAG_ACTION, TAG_WEIGHTS = 5, 6


def recv(comm, source, tag):
    """Blocking receive that sleeps while waiting. A plain comm.recv busy-waits in Open MPI,
    so 9 processes on one laptop would fight over the CPU."""
    while not comm.iprobe(source=source, tag=tag):
        time.sleep(0.00005)
    return comm.recv(source=source, tag=tag)


def parse_args():
    p = argparse.ArgumentParser(description='Distributed IA2C training, one MPI process per car')
    p.add_argument('--seed', type=int, default=12, help='training seed (same meaning as in the configs)')
    p.add_argument('--mode', choices=['independent', 'consensus'], default='independent')
    p.add_argument('--clip', choices=['global', 'local'], default='global')
    p.add_argument('--steps', type=float, default=6e4, help='training steps')
    p.add_argument('--out', required=True, help='run folder (data/, model/best/)')
    p.add_argument('--runs', default=pm.DEFAULT_RUNS)
    p.add_argument('--match-central', action='store_true',
                   help='test mode for check_train.py: both --central-init and --central-rng, so the result '
                        'must equal main.py train')
    p.add_argument('--central-init', action='store_true', help='start from the central code\'s initial weights')
    p.add_argument('--central-rng', action='store_true',
                   help='use the central trainer\'s random numbers for actions (handed out by the road)')
    return p.parse_args()


def load_config(args):
    config = pm.read_config(os.path.join(args.runs, pm.RUN_DIRS['ia2c']))
    config['ENV_CONFIG']['seed'] = str(args.seed)
    config['TRAIN_CONFIG']['total_step'] = str(int(args.steps))
    return config


# ---------------------------------------------------------------- car processes
def car_main(comm, cars, rank, args, config):
    import torch
    import torch.nn as nn
    from agents.policies import LstmPolicy
    from agents.utils import OnPolicyBuffer
    torch.set_num_threads(1)

    i, n = rank - 1, comm.Get_size() - 1
    env = CACCEnv(config['ENV_CONFIG'])            # constants only: targets, OVM, neighbour graph
    mc = config['MODEL_CONFIG']
    nbrs = [int(j) for j in np.where(env.neighbor_mask[i] == 1)[0]]
    n_step = mc.getint('batch_size')
    torch.manual_seed(1000 * args.seed + i)
    rng = np.random.default_rng([args.seed, rank])
    policy = LstmPolicy(env.n_s_ls[i], env.n_a_ls[i], len(nbrs), n_step,
                        n_fc=mc.getint('num_fc'), n_lstm=mc.getint('num_lstm'), name=str(i))
    if args.central_init:
        from agents.models import IA2C          # build the central model exactly as main.py does
        central = IA2C(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       int(args.steps), mc, args.seed)
        policy.load_state_dict(central.policy[i].state_dict())
    optimizer = torch.optim.RMSprop(policy.parameters(), mc.getfloat('lr_init'),
                                    eps=mc.getfloat('rmsp_epsilon'), alpha=mc.getfloat('rmsp_alpha'))
    buffer = OnPolicyBuffer(mc.getfloat('gamma'), env.coop_gamma, env.distance_mask[i])
    e_coef, v_coef = mc.getfloat('entropy_coef'), mc.getfloat('value_coef')
    max_grad, reward_norm = mc.getfloat('max_grad_norm'), mc.getfloat('reward_norm')
    stats = {'updates': 0, 'consensus_bytes': 0}

    def exchange(tag, payload, peers):
        """Send payload to each peer, then receive one message from each (blocking, in order).
        Training has no faults, so plain MPI point-to-point is enough; MPI keeps messages
        between two ranks with the same tag in order."""
        reqs = [comm.isend(payload, dest=j + 1, tag=tag) for j in peers]
        got = [recv(comm, j + 1, tag) for j in peers]
        MPI.Request.Waitall(reqs)
        return got

    def observe(s):
        """V2V rounds 1 and 2: front car's speed, then neighbours' observations."""
        req = comm.isend(s['v'], dest=rank + 1, tag=pm.TAG_BEACON) if i < n - 1 else None
        v_lead = recv(comm, pm.ROAD if i == 0 else rank - 1, pm.TAG_BEACON)
        if req is not None:
            req.wait()
        own = pm.local_state(env, s['v'], s['h'], s['u'], v_lead)
        return np.concatenate([own] + exchange(pm.TAG_STATE, own, nbrs)), v_lead

    def neighbour_actions(a):
        """V2V round 3 (training only): the critic needs the neighbours' actions."""
        return np.array(exchange(TAG_ACTION, a, nbrs))

    def sample(pi, u):
        if u is None:
            return int(rng.choice(len(pi), p=pi / pi.sum()))
        cdf = pi.astype(np.float64).cumsum()     # what np.random.choice does with the road's number u
        cdf /= cdf[-1]
        return int(cdf.searchsorted(u, side='right'))

    def update(R):
        obs, nas, acts, dones, Rs, Advs = buffer.sample_transition(R, 0)
        optimizer.zero_grad()
        policy.backward(obs, nas, acts, dones, Rs, Advs, e_coef, v_coef)
        if max_grad > 0 and args.clip == 'local':
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
        elif max_grad > 0:
            # same rule as torch.nn.utils.clip_grad_norm_ over all 8 networks, done distributed:
            # every car shares the sizes of its gradients (one number per weight tensor, 10 floats),
            # all cars get everyone's, and each scales its own gradient by the same factor
            grads = [p.grad for p in policy.parameters() if p.grad is not None]
            mine = [float(x) for x in torch._foreach_norm(grads, 2)]
            everyone = [x for car in cars.allgather(mine) for x in car]          # in car order
            total = torch.linalg.vector_norm(torch.tensor(everyone, dtype=torch.float32), 2)
            torch._foreach_mul_(grads, torch.clamp(max_grad / (total + 1e-6), max=1.0))
        optimizer.step()
        stats['updates'] += 1
        if args.mode == 'consensus':
            # share knowledge with the neighbours: average the LSTM weights (self + neighbours)
            mine = [p.detach().numpy().copy() for p in policy.lstm_layer.parameters()]
            theirs = exchange(TAG_WEIGHTS, mine, nbrs)
            with torch.no_grad():
                for k, p in enumerate(policy.lstm_layer.parameters()):
                    p.copy_(torch.from_numpy((mine[k] + sum(w[k] for w in theirs)) / (1 + len(theirs))))
            stats['consensus_bytes'] += len(nbrs) * sum(w.size for w in mine) * 4

    comm.Barrier()
    pending, done_prev = None, True
    while True:
        s = recv(comm, pm.ROAD, pm.TAG_SENSOR)
        kind = s['kind']
        if kind == 'stop':
            break
        if kind == 'save':                          # new best: send my weights to the road
            comm.send({k: v.detach().clone() for k, v in policy.state_dict().items()}, dest=pm.ROAD, tag=pm.TAG_CMD)
            continue
        if s['phase'] == 'train' and pending is not None:
            ob_p, na_p, a_p, v_p = pending
            r = s['r'] / reward_norm if reward_norm > 0 else s['r']
            buffer.add_transition(ob_p, na_p, a_p, r, v_p, s['done'])
            pending = None
        if kind == 'end':                           # episode over
            if s['phase'] == 'train' and buffer.obs:
                update(0.0)
            continue
        t = s['t']
        if s['first']:
            policy._reset()
            done_prev = True
        ob, v_lead = observe(s)
        if s['phase'] == 'train':
            if len(buffer.obs) == n_step:               # batch full: bootstrap from the next state, then learn
                pi_b = policy.forward(ob, done_prev)
                a_b = sample(pi_b, s.get('u_boot'))
                R = float(policy.forward(ob, done_prev, neighbour_actions(a_b), 'v'))
                update(R)
            pi = policy.forward(ob, done_prev)
            a = sample(pi, s.get('u_act'))
            na = neighbour_actions(a)
            v = float(policy.forward(ob, done_prev, na, 'v'))
            pending = (ob, na, a, v)
        else:                                       # test episode: deterministic, no learning
            a = int(np.argmax(policy.forward(ob, done_prev)))
        done_prev = False
        alpha, beta = env.a_map[a]
        comm.send({'t': t, 'u': env.ovm.get_accel(s['v'], v_lead, s['h'], alpha, beta), 'a': a},
                  dest=pm.ROAD, tag=pm.TAG_CMD)
    comm.gather(stats, root=pm.ROAD)


# ---------------------------------------------------------------- road process
def road_main(comm, args, config):
    import torch
    n = comm.Get_size() - 1
    env = pm.RoadEnv(config['ENV_CONFIG'])
    assert env.n_agent == n, 'need %d processes' % (env.n_agent + 1)
    total = int(args.steps)
    n_step = config['MODEL_CONFIG'].getint('batch_size')
    for sub in ('data', os.path.join('model', 'best')):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    with open(os.path.join(args.out, 'data', 'ia2c_catchup.ini'), 'w') as f:
        config.write(f)
    log = open(os.path.join(args.out, 'data', 'train_log.jsonl'), 'w')

    def episode(phase):
        env.train_mode = phase == 'train'
        env.reset() if phase == 'train' else env.reset(test_ind=-1)
        t, r, done, rewards = 0, None, True, []
        while True:
            draws = {}
            if args.central_rng and phase == 'train':   # the central trainer's np.random draws, in its order
                if t > 0 and t % n_step == 0:
                    draws['u_boot'] = np.random.random_sample(n)
                draws['u_act'] = np.random.random_sample(n)
            for i in range(n):
                msg = {'kind': 'step', 'phase': phase, 't': t, 'first': t == 0, 'r': r, 'done': done,
                       'v': float(env.vs_cur[i]), 'h': float(env.hs_cur[i]), 'u': float(env.us_cur[i])}
                msg.update({k: float(v[i]) for k, v in draws.items()})
                comm.send(msg, dest=i + 1, tag=pm.TAG_SENSOR)
            comm.send(float(env.v0s[t]), dest=1, tag=pm.TAG_BEACON)          # the leader's beacon
            u, acts = np.zeros(n), [0] * n
            for i in range(n):
                msg = recv(comm, i + 1, pm.TAG_CMD)
                u[i], acts[i] = msg['u'], msg['a']
            env.u_cmd = u
            _, r, done, g = env.step(acts)        # r: the global reward (coop_gamma = -1)
            rewards.append(g)
            t += 1
            if done:
                for i in range(n):
                    comm.send({'kind': 'end', 'phase': phase, 'r': r, 'done': True}, dest=i + 1, tag=pm.TAG_SENSOR)
                return t, float(np.mean(rewards))

    def save(folder, step):
        """Collect every car's weights into one checkpoint in the original format."""
        for i in range(n):
            comm.send({'kind': 'save'}, dest=i + 1, tag=pm.TAG_SENSOR)
        parts = [recv(comm, i + 1, pm.TAG_CMD) for i in range(n)]
        combined = {'%d.%s' % (i, k): v for i, part in enumerate(parts) for k, v in part.items()}
        torch.save({'global_step': step, 'model_state_dict': combined, 'optimizer_state_dict': {}},
                   os.path.join(folder, 'checkpoint-%d.pt' % step))

    comm.Barrier()
    step, history, best, t0 = 0, [], -np.inf, time.perf_counter()
    while step < total:
        n_train, train_r = episode('train')
        step += n_train
        _, test_r = episode('test')
        history.append(test_r)
        rec = {'step': step, 'train_reward': train_r, 'test_reward': test_r, 'minutes': (time.perf_counter() - t0) / 60}
        if len(history) >= 5 and np.mean(history[-5:]) > best:
            best = float(np.mean(history[-5:]))
            best_dir = os.path.join(args.out, 'model', 'best')
            for f in os.listdir(best_dir):
                os.remove(os.path.join(best_dir, f))
            save(best_dir, step)
            rec['new_best'] = best
        log.write(json.dumps(rec) + '\n')
        log.flush()
        print('step %6d  test reward %8.1f  %s' % (step, test_r, '<- best %.1f' % best if 'new_best' in rec else ''),
              flush=True)
    save(os.path.join(args.out, 'model'), step)      # final weights, like main.py train
    for i in range(n):
        comm.send({'kind': 'stop'}, dest=i + 1, tag=pm.TAG_SENSOR)
    stats = comm.gather(None, root=pm.ROAD)[1:]
    summary = {'mode': args.mode, 'clip': args.clip, 'seed': args.seed, 'steps': step, 'best_rolling_test': best,
               'updates_per_car': stats[0]['updates'],
               'consensus_mb_total': round(sum(s['consensus_bytes'] for s in stats) / 1e6, 1),
               'minutes': round((time.perf_counter() - t0) / 60, 1)}
    log.write(json.dumps({'summary': summary}) + '\n')
    log.close()
    print('done:', json.dumps(summary), flush=True)


def main():
    args = parse_args()
    if args.match_central:
        args.central_init = args.central_rng = True
    comm = MPI.COMM_WORLD
    config = load_config(args)
    cars = comm.Split(1 if comm.Get_rank() == pm.ROAD else 0)   # the 8 car processes, without the road
    if comm.Get_rank() == pm.ROAD:
        road_main(comm, args, config)
    else:
        car_main(comm, cars, comm.Get_rank(), args, config)


if __name__ == '__main__':
    main()
