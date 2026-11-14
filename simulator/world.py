"""
Platoon world for the simulator: the CACC physics of envs/cacc_env.py, except that every value
a car gets from another vehicle arrives through the V2V network layer (simulator/network.py).

What each car knows:
  * measured by itself: its own speed and acceleration, and its radar gap to the car in front
  * received over V2V: the speed of the car in front (a "beacon"), and for the learned
    controllers also their neighbours' observations (IA2C) or NeurComm messages
"""
import numpy as np

from envs.cacc_env import CACCEnv, VDIFF
from simulator.network import LEADER

MAX_STEPS = 6000   # 10 minutes of driving per run (training episodes are 60 s)


class NetworkedCACC(CACCEnv):
    """CACCEnv whose controllers only see local measurements plus V2V messages.
    With a perfect network it reproduces CACCEnv exactly (checked by simulator/check.py)."""

    def __init__(self, config, network, comm, n_hidden=0):
        self.net = network
        self.comm = comm            # 'beacon' (fixed controller), 'state' (IA2C) or 'neurcomm'
        self.n_hidden = n_hidden    # NeurComm LSTM size, only used to count message bytes
        self.rx_vlead = None
        super().__init__(config)

    def reset(self, gui=False, test_ind=-1):
        state = super().reset(gui=gui, test_ind=test_ind)
        # allow runs longer than the 60 s training episode: the leader keeps its last speed
        v0s = np.full(MAX_STEPS + 1, self.v0s[-1])
        v0s[:len(self.v0s)] = self.v0s
        self.v0s = v0s
        self.T = MAX_STEPS
        return state

    # ---------------------------------------------------------------- V2V round
    def _exchange(self):
        """One communication round per 0.1 s step."""
        net, n = self.net, self.n_agent
        net.begin_tick()
        # beacons: every vehicle sends its current speed to the car behind it
        net.send(LEADER, 0, 'beacon', self.v0s[self.t], 1)
        for i in range(n - 1):
            net.send(i, i + 1, 'beacon', self.vs_cur[i], 1)
        self.rx_vlead = [net.recv(i, LEADER if i == 0 else i - 1, 'beacon') for i in range(n)]
        own = [self._local_state(i) for i in range(n)]
        if self.comm != 'beacon':
            # IA2C: neighbours share their 5-number observation.
            # NeurComm: observation + latest policy + LSTM hidden state. The policy and hidden
            # parts are still computed inside the shared policy network; they are counted here
            # so the byte numbers are right.
            n_floats = 5 if self.comm == 'state' else 5 + self.n_a + self.n_hidden
            for i in range(n):
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    net.send(int(j), i, self.comm, own[j], n_floats)
        return own

    def _local_state(self, i):
        """The same 5 numbers as CACCEnv._get_veh_state, with the front car's speed from V2V."""
        v_lead = self.rx_vlead[i]
        v = self.vs_cur[i]
        h = self.hs_cur[i]
        v_state = (v - self.v_star) / self.v_star
        vdiff_state = np.clip((v_lead - v) / VDIFF, -2, 2)
        vhdiff_state = np.clip((self.ovm.get_vh(h) - v) / VDIFF, -2, 2)
        h_state = (h + (v_lead - v) * self.dt - self.h_star) / self.h_star
        u_state = self.us_cur[i] / self.u_max
        return np.array([v_state, vdiff_state, vhdiff_state, h_state, u_state])

    def _get_state(self):
        own = self._exchange()
        state = []
        for i in range(self.n_agent):
            cur = [own[i]]
            if self.agent.startswith('ia2c'):
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    cur.append(self.net.recv(i, int(j), self.comm))
            state.append(np.concatenate(cur))
        return state

    def _get_accel(self, i, alpha, beta):
        # the OVM's speed-matching term uses the front car's speed as received over V2V
        return self.ovm.get_accel(self.vs_cur[i], self.rx_vlead[i], self.hs_cur[i], alpha, beta)


class Leader:
    """The leading vehicle. It follows the scenario's speed profile until someone presses
    'Leader brakes', then it brakes, holds a lower speed, and returns to cruising speed."""
    BRAKE_TO = 8.0        # m/s
    BRAKE_DECEL = -2.0    # m/s^2
    HOLD_S = 4.0
    RECOVER_ACCEL = 1.0   # m/s^2

    def __init__(self, dt, v_cruise):
        self.dt = dt
        self.v_cruise = v_cruise
        self.reset(v_cruise)

    def reset(self, v0):
        self.v = v0
        self.u = 0.0            # acceleration over the last step, for the brake lights
        self.segments = []
        self.override = False   # after a brake event the leader ignores the scripted profile

    @property
    def phase(self):
        if not self.segments:
            return 'cruising'
        accel = self.segments[0][0]
        return 'braking' if accel < 0 else ('holding speed' if accel == 0 else 'speeding up')

    def brake(self):
        n_down = max(0, int(round((self.v - self.BRAKE_TO) / (-self.BRAKE_DECEL * self.dt))))
        n_hold = int(round(self.HOLD_S / self.dt))
        n_up = int(round((self.v_cruise - self.BRAKE_TO) / (self.RECOVER_ACCEL * self.dt)))
        self.segments = [[self.BRAKE_DECEL, n_down], [0.0, n_hold], [self.RECOVER_ACCEL, n_up]]
        self.segments = [s for s in self.segments if s[1] > 0]
        self.override = True

    def next_speed(self, scripted_next):
        v_before = self.v
        self._advance(scripted_next)
        self.u = (self.v - v_before) / self.dt
        return self.v

    def _advance(self, scripted_next):
        if self.segments:
            accel, _ = self.segments[0]
            self.v = float(np.clip(self.v + accel * self.dt, 0.0, 30.0))
            self.segments[0][1] -= 1
            if self.segments[0][1] == 0:
                self.segments.pop(0)
                if not self.segments:
                    self.v = self.v_cruise
        elif self.override:
            self.v = self.v_cruise
        else:
            self.v = scripted_next


class Lane:
    """One platoon (8 cars behind a leader) driven by one controller, with its own network."""

    def __init__(self, key, label, env, controller, network):
        self.key, self.label = key, label
        self.env, self.controller, self.net = env, controller, network

    def reset(self, seed, scenario):
        env = self.env
        env.name = scenario
        env.train_mode = False
        env.init_test_seeds([seed])
        self.net.reset()
        self.obs = env.reset(test_ind=0)
        self.done = True               # resets the LSTM state on the first step, like in evaluation
        self.controller.reset()
        self.x_leader = 0.0
        self.status = 'ok'
        self.collision_t = None
        self.rewards = []
        self.last_reward = 0.0
        self.last_action = None

    def links(self):
        """Who sends what to whom in one round, e.g. [[-1, 0, 'beacon', 4], ...]."""
        seen, out = set(), []
        for src, dst, kind, n_bytes in self.net.sent:
            if (src, dst, kind) not in seen:
                seen.add((src, dst, kind))
                out.append([int(src), int(dst), kind, int(n_bytes)])
        return out

    def step(self, v0_next):
        if self.status != 'ok':
            return
        env = self.env
        t = env.t
        env.v0s[t + 1] = v0_next
        action = self.controller.act(self.obs, self.done, env)
        self.obs, _, self.done, reward = env.step(action)
        self.x_leader += 0.5 * env.dt * (float(env.v0s[t]) + float(v0_next))
        self.last_action = action
        self.last_reward = float(reward)
        self.rewards.append(self.last_reward)
        if env.collision:
            self.status = 'collision'
            self.collision_t = round(env.t * env.dt, 1)
        elif self.done:
            self.status = 'ended'

    def frame(self):
        env = self.env
        rel = -np.cumsum(env.hs_cur)      # car positions behind the leader [m]
        return {
            'status': self.status, 'ct': self.collision_t,
            'xl': round(self.x_leader, 2),
            'x': [round(float(x), 2) for x in rel],
            'h': [round(float(h), 2) for h in env.hs_cur],
            'v': [round(float(v), 2) for v in env.vs_cur],
            'u': [round(float(u), 2) for u in env.us_cur],
            'a': [] if self.last_action is None else [int(a) for a in self.last_action],
            'r': round(self.last_reward, 1),
            'ravg': round(float(np.mean(self.rewards)), 1) if self.rewards else None,
            'mk': len(self.net.sent), 'bk': self.net.tick_bytes,
            'mt': self.net.msgs_total, 'bt': self.net.bytes_total,
        }
