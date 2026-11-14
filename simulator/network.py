"""
V2V network layer of the platoon simulator.

Every message between vehicles goes through Network.send(), so the dashboard can show the
traffic and count messages and bytes. Phase 1 delivers every message immediately (a perfect
network). Loss, delay, cut links and crashed nodes will be added here, without changing the
vehicles or the controllers.
"""
from collections import defaultdict

LEADER = -1        # node id of the leading vehicle (it broadcasts but is not an RL agent)
FLOAT_BYTES = 4    # values go over the wire as float32


class Network:
    def __init__(self):
        self.reset()

    def reset(self):
        self.inbox = defaultdict(dict)   # dst -> {(src, kind): payload}
        self.sent = []                   # this tick's messages: (src, dst, kind, n_bytes)
        self.msgs_total = 0
        self.bytes_total = 0

    def begin_tick(self):
        self.sent = []

    def send(self, src, dst, kind, payload, n_floats):
        n_bytes = n_floats * FLOAT_BYTES
        self.sent.append((src, dst, kind, n_bytes))
        self.msgs_total += 1
        self.bytes_total += n_bytes
        self.inbox[dst][(src, kind)] = payload     # perfect network: delivered at once

    def recv(self, dst, src, kind):
        return self.inbox[dst][(src, kind)]

    @property
    def tick_bytes(self):
        return sum(m[3] for m in self.sent)
