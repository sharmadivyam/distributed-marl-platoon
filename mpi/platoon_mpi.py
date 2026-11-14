"""
The CACC platoon on MPI: every car is its own MPI process.

    mpirun --oversubscribe -np 9 python mpi/platoon_mpi.py --controller ia2c   # 8 cars + the road

Rank 0 is the road. It moves the cars (the physics) and drives the leading vehicle.
Rank i (1..N) is car i. It only knows its own sensor readings (speed, acceleration and radar gap,
sent by the road) and what its neighbouring cars send it over V2V. It runs its own policy and
sends its acceleration command back to the road. Cars talk to each other directly, rank to rank.

Fault injection on the V2V links:
    --drop 0.2                      each V2V message is lost with probability 0.2
    --delay 3                       V2V messages arrive 3 steps (0.3 s) late
    --kill-car 4 --kill-at 30       car 4's process goes silent at t = 30 s (crash-stop)
When a message does not arrive in time, the car falls back to the last value it received
(--fallback last) or estimates the front car's speed from its own radar (--fallback radar).

Rank 0 writes every step to a log (--log) that the browser dashboard can replay:
    python -m simulator.server --replay mpi/logs/<run>.jsonl
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
from mpi4py import MPI

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
logging.disable(logging.INFO)

from envs.cacc_env import CACCEnv, VDIFF  # noqa: E402
from simulator.world import Leader  # noqa: E402

# same values as simulator/controllers.py, repeated here so the fixed controller does not need
# PyTorch (only IA2C cars load it); this keeps 32-car runs light
DEFAULT_RUNS = os.path.join(REPO, 'demo', 'runs')
RUN_DIRS = {'ia2c': 'ia2c_catchup', 'neurcomm': 'ma2c_nc_catchup'}
FIXED_SETTING = (0.5, 0.5)


def read_config(run_dir):
    import configparser
    import glob
    paths = glob.glob(os.path.join(run_dir, 'data', '*.ini'))
    if not paths:
        raise SystemExit('no config in %s/data (train it first, see demo/DEMO_GUIDE.md)' % run_dir)
    config = configparser.ConfigParser()
    config.read(paths[0])
    return config

ROAD = 0
TAG_SENSOR, TAG_CMD, TAG_BEACON, TAG_STATE = 1, 2, 3, 4
FLOAT_BYTES = 4
LEADER_ID = -1        # node id of the leader in logs (it lives on the road rank)


def parse_args():
    p = argparse.ArgumentParser(description='CACC platoon with one MPI process per car')
    p.add_argument('--controller', choices=['fixed', 'ia2c'], default='ia2c')
    p.add_argument('--seed', type=int, default=100000, help='starting position (a report test seed)')
    p.add_argument('--scenario', choices=['catchup', 'slowdown'], default='catchup')
    p.add_argument('--steps', type=int, default=600, help='0.1 s steps to simulate (600 = 60 s)')
    p.add_argument('--brake-at', type=float, default=None, help='leader brakes at this time [s]')
    p.add_argument('--drop', type=float, default=0.0, help='probability that a V2V message is lost')
    p.add_argument('--delay', type=int, default=0, help='V2V messages arrive this many steps late')
    p.add_argument('--kill-car', type=int, default=None, help='car number (1..N) whose process crashes')
    p.add_argument('--kill-at', type=float, default=30.0, help='time of the crash [s]')
    p.add_argument('--fallback', choices=['last', 'radar'], default='last',
                   help='what a car uses when the front car\'s message is missing')
    p.add_argument('--timeout-ms', type=float, default=50.0,
                   help='how long a car waits for a neighbour\'s message before falling back')
    p.add_argument('--crash-timeout-ms', type=float, default=2000.0,
                   help='how long the road waits for a car\'s command before declaring it crashed')
    p.add_argument('--cars', type=int, default=None, help='platoon size (fixed controller only)')
    p.add_argument('--realtime', action='store_true', help='run at 10 steps per second')
    p.add_argument('--runs', default=DEFAULT_RUNS)
    p.add_argument('--log', default=None, help='JSON-lines log for the dashboard (written by rank 0)')
    return p.parse_args()


def load_config(args):
    run = RUN_DIRS['ia2c'] if args.controller == 'ia2c' else RUN_DIRS['neurcomm']
    run_dir = os.path.join(args.runs, run)
    config = read_config(run_dir)
    if args.cars is not None:
        if args.controller != 'fixed' and args.cars != config['ENV_CONFIG'].getint('n_vehicle'):
            raise SystemExit('the trained IA2C policy only works with %s cars' % config['ENV_CONFIG']['n_vehicle'])
        config['ENV_CONFIG']['n_vehicle'] = str(args.cars)
    return config, run_dir


# ---------------------------------------------------------------- V2V layer
class V2V:
    """One rank's side of the V2V network: sends with fault injection, receives with a timeout."""

    def __init__(self, comm, rank, args):
        self.comm = comm
        self.drop, self.delay = args.drop, args.delay
        self.rng = np.random.default_rng([args.seed, rank])
        self.queue = []       # (send_tick, dest, tag, payload) held back by --delay
        self.pending = []     # MPI requests of messages in flight
        self.newest = {}      # (src, tag) -> newest message received
        self.new_tick()

    def new_tick(self):
        self.sent = self.bytes = self.dropped = self.stale = 0
        self.lost = []

    def send(self, t, dest, tag, payload, n_floats, src_id, dst_id):
        if self.drop > 0 and self.rng.random() < self.drop:
            self.dropped += 1
            self.lost.append([src_id, dst_id])
            return
        self.sent += 1
        self.bytes += n_floats * FLOAT_BYTES
        self.queue.append((t + self.delay, dest, tag, payload))
        self.flush(t)

    def flush(self, t):
        due = [m for m in self.queue if m[0] <= t]
        self.queue = [m for m in self.queue if m[0] > t]
        for _, dest, tag, payload in due:
            self.pending.append(self.comm.isend(payload, dest=dest, tag=tag))
        self.pending = [r for r in self.pending if not r.Test()]

    def recv(self, t, src, tag, timeout):
        """The newest message from src. Waits up to `timeout` seconds for the one sent at tick t;
        returns (message or None, stale) where stale means it is older than tick t."""
        deadline = time.perf_counter() + timeout
        while True:
            while self.comm.iprobe(source=src, tag=tag):
                msg = self.comm.recv(source=src, tag=tag)
                old = self.newest.get((src, tag))
                if old is None or msg['t'] >= old['t']:
                    self.newest[(src, tag)] = msg
            cur = self.newest.get((src, tag))
            if cur is not None and cur['t'] == t:
                return cur, False
            if time.perf_counter() > deadline:
                self.stale += 1
                return cur, True
            time.sleep(0.0001)


# ---------------------------------------------------------------- car processes
def local_state(env, v, h, u, v_lead):
    """Same 5 numbers as CACCEnv._get_veh_state, from this car's own data plus V2V."""
    v_state = (v - env.v_star) / env.v_star
    vdiff_state = np.clip((v_lead - v) / VDIFF, -2, 2)
    vhdiff_state = np.clip((env.ovm.get_vh(h) - v) / VDIFF, -2, 2)
    h_state = (h + (v_lead - v) * env.dt - env.h_star) / env.h_star
    u_state = u / env.u_max
    return np.array([v_state, vdiff_state, vhdiff_state, h_state, u_state])


def car_main(comm, rank, args, config, run_dir):
    i = rank - 1                              # car index 0..N-1 (car number i+1 on screen)
    n = comm.Get_size() - 1
    env = CACCEnv(config['ENV_CONFIG'])       # only for the constants: targets, OVM, neighbour graph
    neighbours = [int(j) for j in np.where(env.neighbor_mask[i] == 1)[0]]
    if args.controller == 'ia2c':
        from simulator.controllers import load_model          # PyTorch, only needed for IA2C
        policy = load_model(env, config, run_dir).policy[i]   # this car's own network only
        policy._reset()
    else:
        fixed_action = env.a_map.index(FIXED_SETTING)
    v2v = V2V(comm, rank, args)
    timeout = args.timeout_ms / 1000.0
    comm.gather(MPI.Get_processor_name(), root=ROAD)   # tell the road which machine I run on
    comm.Barrier()       # start together: no car may look "crashed" just because it loaded slowly
    kill_tick = int(round(args.kill_at / env.dt)) if args.kill_car == i + 1 else None
    done, dead, h_prev, own = True, False, None, np.zeros(5)

    while True:
        sensor = comm.recv(source=ROAD, tag=TAG_SENSOR)
        if sensor['stop']:
            break
        t = sensor['t']
        if kill_tick is not None and t >= kill_tick:
            dead = True                       # crash-stop: this process stays alive but goes silent
        if dead:
            continue
        v2v.new_tick()
        v2v.flush(t)
        v, h, u = sensor['v'], sensor['h'], sensor['u']

        # V2V round 1: beacon (my speed) to the car behind, beacon from the car in front
        if i < n - 1:
            v2v.send(t, rank + 1, TAG_BEACON, {'t': t, 'v': v}, 1, i, i + 1)
        msg, stale = v2v.recv(t, ROAD if i == 0 else rank - 1, TAG_BEACON, timeout)
        if not stale:
            v_lead = msg['v']
        elif args.fallback == 'radar' and h_prev is not None:
            v_lead = v + (h - h_prev) / env.dt  # front car's speed estimated from the radar gap
        else:
            v_lead = msg['v'] if msg is not None else v
        h_prev = h
        own = local_state(env, v, h, u, v_lead)

        # V2V round 2 (IA2C only): share observations with both neighbours, then act
        if args.controller == 'ia2c':
            for j in neighbours:
                v2v.send(t, j + 1, TAG_STATE, {'t': t, 's': own}, 5, i, j)
            parts = [own]
            for j in neighbours:
                msg, _ = v2v.recv(t, j + 1, TAG_STATE, timeout)
                parts.append(msg['s'] if msg is not None else np.zeros(5))
            pi = policy.forward(np.concatenate(parts), done)
            action = int(np.argmax(pi))
        else:
            action = fixed_action
        done = False

        alpha, beta = env.a_map[action]
        u_cmd = env.ovm.get_accel(v, v_lead, h, alpha, beta)
        comm.send({'t': t, 'u': u_cmd, 'a': action, 'sent': v2v.sent, 'bytes': v2v.bytes,
                   'dropped': v2v.dropped, 'lost': v2v.lost, 'stale': v2v.stale},
                  dest=ROAD, tag=TAG_CMD)


# ---------------------------------------------------------------- road process
class RoadEnv(CACCEnv):
    """CACC physics only: the cars' accelerations come from their own commands."""
    u_cmd = None

    def _get_state(self):
        return []

    def _get_accel(self, i, alpha, beta):
        return self.u_cmd[i]


def links_for(controller, n):
    links = [[LEADER_ID, 0, 'beacon', FLOAT_BYTES]] + [[i, i + 1, 'beacon', FLOAT_BYTES] for i in range(n - 1)]
    if controller == 'ia2c':
        for i in range(n):
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    links.append([j, i, 'state', 5 * FLOAT_BYTES])
    return links


def span(ranks):
    return '%d-%d' % (ranks[0], ranks[-1]) if len(ranks) > 1 and ranks[-1] - ranks[0] == len(ranks) - 1 \
        else ','.join(map(str, ranks))


def wait_command(comm, src, t, timeout):
    """The car's command for tick t, or None if it does not arrive in time (crashed car)."""
    deadline = time.perf_counter() + timeout
    while True:
        while comm.iprobe(source=src, tag=TAG_CMD):
            msg = comm.recv(source=src, tag=TAG_CMD)
            if msg['t'] == t:
                return msg
        if time.perf_counter() > deadline:
            return None
        time.sleep(0.0001)


def road_main(comm, args, config):
    n = comm.Get_size() - 1
    hosts = comm.gather(MPI.Get_processor_name(), root=ROAD)   # machine of every rank (waits for all cars)
    by_host = {}
    for r, h in enumerate(hosts):
        by_host.setdefault(h.split('.')[0], []).append(r)
    placement = ', '.join('ranks %s on %s' % (span(rs), h) for h, rs in by_host.items())
    env = RoadEnv(config['ENV_CONFIG'])
    if env.n_agent != n:
        raise SystemExit('need %d MPI processes (1 road + %d cars), got %d' % (env.n_agent + 1, env.n_agent, n + 1))
    env.name = args.scenario
    env.train_mode = False
    env.init_test_seeds([args.seed])
    env.reset(test_ind=0)
    # allow runs longer than the 60 s training episode
    v0s = np.full(max(args.steps, env.T) + 2, env.v0s[-1])
    v0s[:len(env.v0s)] = env.v0s
    env.v0s, env.T = v0s, max(args.steps, env.T)
    script = env.v0s.copy()
    leader = Leader(env.dt, env.v_star)
    leader.reset(float(script[0]))
    brake_tick = None if args.brake_at is None else int(round(args.brake_at / env.dt))

    v2v = V2V(comm, ROAD, args)
    alive, crashed_at = [True] * n, {}
    rewards, x_leader = [], 0.0
    totals = {'sent': 0, 'bytes': 0, 'dropped': 0, 'stale': 0}
    log = open(args.log, 'w') if args.log else None
    label = '%s on MPI: %d car processes + 1 road process' % ('IA2C' if args.controller == 'ia2c' else 'Fixed controller', n)
    faults = []
    if args.drop:
        faults.append('%d%% messages lost' % round(args.drop * 100))
    if args.delay:
        faults.append('%.1f s delay' % (args.delay * env.dt))
    if args.kill_car:
        faults.append('car %d crashes at %.0f s' % (args.kill_car, args.kill_at))
    if args.brake_at is not None:
        faults.append('leader brakes at %.0f s' % args.brake_at)
    if args.drop or args.delay or args.kill_car:
        faults.append('fallback: %s' % ('radar estimate' if args.fallback == 'radar' else 'last value'))
    if len(by_host) > 1:
        faults.insert(0, '%d machines' % len(by_host))
    config_out = {'replay': True, 'seed': args.seed, 'seeds': [args.seed], 'scenario': args.scenario,
                  'n': n, 'dt': env.dt, 'h_star': env.h_star, 'v_star': env.v_star, 'h_min': env.h_min,
                  'a_map': [list(a) for a in env.a_map],
                  'network': 'MPI, ' + (', '.join(faults) if faults else 'no faults'),
                  'lanes': [{'key': 'mpi', 'label': label, 'comm': 'state' if args.controller == 'ia2c' else 'beacon',
                             'links': links_for(args.controller, n)}]}

    def frame(t, tick_stats, status, collision_t):
        rel = -np.cumsum(env.hs_cur)
        return {'t': round(t * env.dt, 2), 'tick': t, 'playing': True, 'speed': 1.0,
                'vl': round(leader.v, 2), 'ul': round(leader.u, 2), 'phase': leader.phase,
                'lanes': [{'status': status, 'ct': collision_t, 'xl': round(x_leader, 2),
                           'x': [round(float(x), 2) for x in rel], 'h': [round(float(h), 2) for h in env.hs_cur],
                           'v': [round(float(v), 2) for v in env.vs_cur], 'u': [round(float(u), 2) for u in env.us_cur],
                           'a': tick_stats['a'], 'r': round(rewards[-1], 1) if rewards else 0.0,
                           'ravg': round(float(np.mean(rewards)), 1) if rewards else None,
                           'mk': tick_stats['sent'], 'bk': tick_stats['bytes'], 'mt': totals['sent'], 'bt': totals['bytes'],
                           'drop': tick_stats['dropped'], 'dropt': totals['dropped'], 'lost': tick_stats['lost'],
                           'stale': tick_stats['stale_cars'], 'dead': [i for i in range(n) if not alive[i]]}]}

    if log:
        log.write(json.dumps({'config': config_out}) + '\n')
        log.write(json.dumps({'frame': frame(0, {'a': [], 'sent': 0, 'bytes': 0, 'dropped': 0, 'lost': [],
                                                 'stale_cars': []}, 'ok', None)}) + '\n')
    comm.Barrier()       # wait until every car process has loaded its policy
    status, collision_t, t_start = 'ok', None, time.perf_counter()
    t = 0
    for t in range(args.steps):
        tick_start = time.perf_counter()
        v2v.new_tick()
        v2v.flush(t)
        for i in range(n):
            comm.send({'t': t, 'stop': False, 'v': float(env.vs_cur[i]), 'h': float(env.hs_cur[i]),
                       'u': float(env.us_cur[i])}, dest=i + 1, tag=TAG_SENSOR)
        v2v.send(t, 1, TAG_BEACON, {'t': t, 'v': float(env.v0s[t])}, 1, LEADER_ID, 0)   # leader's beacon

        stats = {'a': [0] * n, 'sent': v2v.sent, 'bytes': v2v.bytes, 'dropped': v2v.dropped,
                 'lost': list(v2v.lost), 'stale_cars': []}
        u_cmd = np.zeros(n)
        for i in range(n):
            if not alive[i]:
                continue
            msg = wait_command(comm, i + 1, t, args.crash_timeout_ms / 1000.0)
            if msg is None:                       # no command: the car's controller has crashed
                alive[i] = False
                crashed_at[i + 1] = round(t * env.dt, 1)
                continue
            u_cmd[i], stats['a'][i] = msg['u'], msg['a']
            stats['sent'] += msg['sent']
            stats['bytes'] += msg['bytes']
            stats['dropped'] += msg['dropped']
            stats['lost'] += msg['lost']
            if msg['stale']:
                stats['stale_cars'].append(i)
                totals['stale'] += msg['stale']
        for key in ('sent', 'bytes', 'dropped'):
            totals[key] += stats[key]

        # physics: a crashed car gets no command and keeps rolling at its current speed
        if brake_tick is not None and t == brake_tick:
            leader.brake()
        v_next = leader.next_speed(float(script[t + 1]))
        env.v0s[t + 1] = v_next
        env.u_cmd = u_cmd
        v0_now = float(env.v0s[t])
        _, _, done, global_reward = env.step(stats['a'])
        x_leader += 0.5 * env.dt * (v0_now + v_next)
        rewards.append(float(global_reward))
        if env.collision:
            status, collision_t = 'collision', round((t + 1) * env.dt, 1)
        if log:
            log.write(json.dumps({'frame': frame(t + 1, stats, status, collision_t)}) + '\n')
        if status != 'ok' or done:
            break
        if args.realtime:
            time.sleep(max(0.0, env.dt - (time.perf_counter() - tick_start)))

    wall = time.perf_counter() - t_start
    for i in range(n):
        comm.send({'stop': True}, dest=i + 1, tag=TAG_SENSOR)
    summary = {'controller': args.controller, 'cars': n, 'processes': n + 1, 'seed': args.seed,
               'placement': placement, 'machines': len(by_host),
               'scenario': args.scenario, 'drop': args.drop, 'delay_steps': args.delay,
               'fallback': args.fallback, 'kill_car': args.kill_car, 'steps': len(rewards),
               'avg_reward': float(np.mean(rewards)), 'collision_t': collision_t,
               'crashed_cars': crashed_at, 'msgs_sent': totals['sent'], 'bytes_sent': totals['bytes'],
               'msgs_dropped': totals['dropped'], 'stale_reads': totals['stale'],
               'wall_s': round(wall, 2), 'ms_per_step': round(1000 * wall / max(1, len(rewards)), 2),
               'rewards': rewards}
    if log:
        log.write(json.dumps({'summary': summary}) + '\n')
        log.close()
    return summary


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    config, run_dir = load_config(args)
    if rank == ROAD:
        s = road_main(comm, args, config)
        print('MPI run: %d processes (%d cars + road), controller=%s, seed=%d, %s'
              % (s['processes'], s['cars'], s['controller'], s['seed'], s['scenario']))
        print('  placement: %s' % s['placement'])
        print('  steps %d, avg reward/step %.2f, collision at %s s, crashed cars %s'
              % (s['steps'], s['avg_reward'], s['collision_t'], s['crashed_cars'] or 'none'))
        print('  V2V: %d messages sent (%d bytes), %d lost, %d stale reads (fallback used)'
              % (s['msgs_sent'], s['bytes_sent'], s['msgs_dropped'], s['stale_reads']))
        print('  wall time %.2f s, %.2f ms per step' % (s['wall_s'], s['ms_per_step']))
        if args.log:
            print('  log: %s' % args.log)
    else:
        car_main(comm, rank, args, config, run_dir)
    comm.Barrier()


if __name__ == '__main__':
    main()
