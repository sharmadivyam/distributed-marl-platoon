"""
Checks for the platoon simulator (run from the repository root):

    python -m simulator.check

1. Equivalence: with a perfect network every platoon must reproduce the original CACCEnv
   evaluation step for step (same rewards, same gaps). Then the simulator shows exactly the
   behaviour that the report measured.
2. Brake test: what each controller does when the leader brakes at 30 s, and the speed of
   one simulation step.
"""
import logging
import time

import numpy as np

from simulator.controllers import (DEFAULT_RUNS, RUN_DIRS, FixedController, LearnedController,
                                   build_lane, load_model, read_config)
from simulator.world import Leader
from envs.cacc_env import CACCEnv

SEED = 100000
STEPS = 600


def reference_rollout(kind, seed=SEED, steps=STEPS):
    """The original evaluation loop (Trainer.perform in test mode) on a plain CACCEnv."""
    import os
    run_dir = os.path.join(DEFAULT_RUNS, RUN_DIRS.get(kind, 'ma2c_nc_catchup'))
    config = read_config(run_dir)
    env = CACCEnv(config['ENV_CONFIG'])
    env.train_mode = False
    env.init_test_seeds([seed])
    if kind == 'fixed':
        ctrl = FixedController(env)
    else:
        ctrl = LearnedController(load_model(env, config, run_dir), kind == 'neurcomm')
    ob, done = env.reset(test_ind=0), True
    ctrl.reset()
    rewards, gaps = [], []
    for _ in range(steps):
        ob, _, done, r = env.step(ctrl.act(ob, done, env))
        rewards.append(r)
        gaps.append(env.hs_cur.copy())
        if done:
            break
    return np.array(rewards), np.array(gaps)


def lane_rollout(lane, n_steps, seed=SEED, scenario='catchup', brake_at=None):
    lane.reset(seed, scenario)
    script = lane.env.v0s.copy()
    leader = Leader(lane.env.dt, lane.env.v_star)
    leader.reset(float(script[0]))
    rewards, gaps = [], []
    for t in range(n_steps):
        if brake_at is not None and t == brake_at:
            leader.brake()
        lane.step(leader.next_speed(float(script[t + 1])))
        if lane.status != 'ok' and lane.status != 'ended':
            break
        rewards.append(lane.last_reward)
        gaps.append(lane.env.hs_cur.copy())
    return np.array(rewards), np.array(gaps)


def main():
    logging.disable(logging.INFO)
    print('1) Equivalence with the original environment (seed %d, %d steps)' % (SEED, STEPS))
    all_ok = True
    for kind in ('fixed', 'ia2c', 'neurcomm'):
        ref_r, ref_h = reference_rollout(kind)
        lane = build_lane(kind)
        sim_r, sim_h = lane_rollout(lane, len(ref_r))
        same = len(ref_r) == len(sim_r) and np.array_equal(ref_r, sim_r) and np.array_equal(ref_h, sim_h)
        all_ok &= same
        print('   %-9s %s   avg reward/step: original %.2f, simulator %.2f   V2V: %d msgs, %d bytes per step'
              % (kind, 'IDENTICAL' if same else 'DIFFERENT', ref_r.mean(), sim_r.mean(),
                 len(lane.net.sent), lane.net.tick_bytes))
    print('   ->', 'all identical' if all_ok else 'MISMATCH, do not trust the simulator yet')

    print('\n2) Leader brakes at 30 s (seed %d, 60 s)' % SEED)
    for kind in ('fixed', 'ia2c', 'neurcomm'):
        lane = build_lane(kind)
        t0 = time.perf_counter()
        r, h = lane_rollout(lane, 600, brake_at=300)
        ms = (time.perf_counter() - t0) / max(1, len(r)) * 1000
        after = h[300:] if len(h) > 300 else h
        print('   %-9s status=%-9s collision_t=%-5s min gap after brake %.1f m   gap error at end %.2f m   %.2f ms/step'
              % (kind, lane.status, lane.collision_t, after.min() if len(after) else float('nan'),
                 np.abs(h[-1] - lane.env.h_star).mean(), ms))


if __name__ == '__main__':
    main()
