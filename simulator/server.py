"""
Platoon simulator: three platoons side by side (fixed controller, IA2C, NeurComm), streamed to
a dashboard in the browser.

    python -m simulator.server --open        # from the repository root

Everything stays on this machine: the server only listens on 127.0.0.1 and uses just the Python
standard library (Server-Sent Events for the live stream, POST /cmd for the controls).
"""
import argparse
import json
import logging
import os
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from simulator.controllers import DEFAULT_RUNS, build_lane
from simulator.world import MAX_STEPS, Leader

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
SEEDS = list(range(100000, 100100, 10))      # the first ten unseen test seeds from the report
SCENARIOS = ('catchup', 'slowdown')
AUTO_BRAKE_S = 30.0
CONTENT_TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'application/javascript; charset=utf-8',
                 '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png'}


class Hub:
    """Sends server-sent events to every open dashboard tab."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clients = set()
        self.last = {}          # last payload per event, replayed to tabs that connect later

    def subscribe(self):
        q = queue.Queue(maxsize=256)
        with self.lock:
            self.clients.add(q)
            for event in ('config', 'frame'):
                if event in self.last:
                    q.put_nowait((event, self.last[event]))
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.discard(q)

    def publish(self, event, payload):
        data = json.dumps(payload, separators=(',', ':'))
        with self.lock:
            self.last[event] = data
            for q in list(self.clients):
                try:
                    q.put_nowait((event, data))
                except queue.Full:
                    pass        # a slow tab skips frames instead of slowing the simulation


class Simulation:
    KINDS = ('fixed', 'ia2c', 'neurcomm')

    def __init__(self, hub, runs_dir):
        self.hub = hub
        self.lock = threading.RLock()
        self.lanes = [build_lane(kind, runs_dir) for kind in self.KINDS]
        env = self.lanes[0].env
        self.dt = env.dt
        self.leader = Leader(env.dt, env.v_star)
        self.playing = False
        self.speed = 1.0
        self.reset(SEEDS[0], 'catchup')

    # ------------------------------------------------------------------ state
    def reset(self, seed, scenario):
        if scenario not in SCENARIOS:
            raise ValueError('unknown scenario %r' % scenario)
        with self.lock:
            self.seed, self.scenario = int(seed), scenario
            for lane in self.lanes:
                lane.reset(self.seed, scenario)
            self.script = self.lanes[0].env.v0s.copy()      # the scenario's leader speed profile
            self.leader.reset(float(self.script[0]))
            self.tick = 0
            self.playing = False
            self.auto_brake_tick = None
            self.hub.publish('config', self.config())
            self.hub.publish('frame', self.frame())

    def config(self):
        env = self.lanes[0].env
        return {'seed': self.seed, 'seeds': SEEDS, 'scenario': self.scenario,
                'n': env.n_agent, 'dt': env.dt, 'h_star': env.h_star, 'v_star': env.v_star,
                'h_min': env.h_min, 'a_map': [list(a) for a in env.a_map],
                'lanes': [{'key': l.key, 'label': l.label, 'comm': l.env.comm, 'links': l.links()}
                          for l in self.lanes]}

    def frame(self):
        return {'t': round(self.tick * self.dt, 2), 'tick': self.tick, 'playing': self.playing,
                'speed': self.speed, 'vl': round(self.leader.v, 2), 'ul': round(self.leader.u, 2),
                'phase': self.leader.phase,
                'lanes': [lane.frame() for lane in self.lanes]}

    # ------------------------------------------------------------------ running
    def step(self):
        with self.lock:
            if self.tick >= MAX_STEPS - 1 or all(l.status != 'ok' for l in self.lanes):
                self.playing = False
                self.hub.publish('frame', self.frame())
                return
            if self.auto_brake_tick is not None and self.tick == self.auto_brake_tick:
                self.leader.brake()
                self.auto_brake_tick = None
            v_next = self.leader.next_speed(float(self.script[self.tick + 1]))
            for lane in self.lanes:
                lane.step(v_next)
            self.tick += 1
            self.hub.publish('frame', self.frame())

    def run(self):
        while True:
            t0 = time.perf_counter()
            if self.playing:
                self.step()
            period = self.dt / self.speed if self.playing else 0.05
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def command(self, msg):
        cmd = msg.get('cmd')
        with self.lock:
            if cmd == 'play':
                self.playing = True
            elif cmd == 'pause':
                self.playing = False
            elif cmd == 'toggle':
                self.playing = not self.playing
            elif cmd == 'reset':
                self.reset(msg.get('seed', self.seed), msg.get('scenario', self.scenario))
            elif cmd == 'brake':
                self.leader.brake()
            elif cmd == 'speed':
                self.speed = min(8.0, max(0.25, float(msg.get('value', 1.0))))
            elif cmd == 'auto':
                # scripted run for demos and recordings: catch-up, then the leader brakes at 30 s
                self.reset(msg.get('seed', self.seed), 'catchup')
                self.auto_brake_tick = int(round(AUTO_BRAKE_S / self.dt))
                self.playing = True
            else:
                return False
            self.hub.publish('frame', self.frame())
            return True


class Replay:
    """Plays back MPI runs (logs written by mpi/platoon_mpi.py) on the same dashboard.
    Several logs are shown as stacked lanes, so runs can be compared side by side."""

    def __init__(self, hub, paths):
        self.hub = hub
        self.lock = threading.RLock()
        self.runs = []
        for path in paths:
            cfg, frames = None, []
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    if 'config' in rec:
                        cfg = rec['config']
                    elif 'frame' in rec:
                        frames.append(rec['frame'])
            if cfg is None or not frames:
                raise ValueError('%s is not an MPI run log' % path)
            self.runs.append((cfg, frames))
        self.dt = self.runs[0][0]['dt']
        self.n_ticks = max(len(frames) for _, frames in self.runs)
        self.speed, self.playing, self.i = 1.0, False, 0
        self.hub.publish('config', self.config())
        self.hub.publish('frame', self.frame())

    def config(self):
        out = dict(self.runs[0][0])
        out['lanes'] = []
        for cfg, _ in self.runs:
            lane = dict(cfg['lanes'][0])
            lane['label'] = '%s  |  %s' % (lane['label'], cfg['network'])
            out['lanes'].append(lane)
        out['network'] = 'Replay of %d MPI run%s' % (len(self.runs), '' if len(self.runs) == 1 else 's')
        return out

    def frame(self):
        frames = [frames[min(self.i, len(frames) - 1)] for _, frames in self.runs]
        # header shows the leader of a run where it is doing something (e.g. braking), if any
        out = dict(next((f for f in frames if f.get('phase', 'cruising') != 'cruising'), frames[0]))
        out.update({'t': round(self.i * self.dt, 2), 'tick': self.i, 'playing': self.playing,
                    'speed': self.speed, 'lanes': [f['lanes'][0] for f in frames]})
        return out

    def step(self):
        with self.lock:
            if self.i >= self.n_ticks - 1:
                self.playing = False
            else:
                self.i += 1
            self.hub.publish('frame', self.frame())

    def run(self):
        while True:
            t0 = time.perf_counter()
            if self.playing:
                self.step()
            period = self.dt / self.speed if self.playing else 0.05
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def command(self, msg):
        cmd = msg.get('cmd')
        with self.lock:
            if cmd in ('play', 'toggle') and not self.playing and self.i >= self.n_ticks - 1:
                self.i = 0            # at the end: Play starts the replay again from 0 s
            if cmd == 'play':
                self.playing = True
            elif cmd == 'pause':
                self.playing = False
            elif cmd == 'toggle':
                self.playing = not self.playing
            elif cmd in ('reset', 'auto'):
                self.i = 0
                self.playing = cmd == 'auto'
                self.hub.publish('config', self.config())
            elif cmd == 'speed':
                self.speed = min(8.0, max(0.25, float(msg.get('value', 1.0))))
            elif cmd != 'brake':      # the leader's actions are already in the recording
                return False
            self.hub.publish('frame', self.frame())
            return True


class Handler(BaseHTTPRequestHandler):
    sim = None
    hub = None

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/stream':
            return self._stream()
        if path in ('/', '/index.html'):
            path = '/static/index.html'
        if path.startswith('/static/'):
            return self._file(path[len('/static/'):])
        self.send_error(404)

    def do_POST(self):
        if self.path != '/cmd':
            return self.send_error(404)
        try:
            length = int(self.headers.get('Content-Length', 0))
            ok = self.sim.command(json.loads(self.rfile.read(length) or b'{}'))
        except (ValueError, TypeError) as err:
            return self._json(400, {'ok': False, 'error': str(err)})
        self._json(200 if ok else 400, {'ok': ok})

    def _json(self, code, obj):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, rel):
        full = os.path.normpath(os.path.join(STATIC, rel))
        if not full.startswith(STATIC + os.sep) or not os.path.isfile(full):
            return self.send_error(404)
        with open(full, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', CONTENT_TYPES.get(os.path.splitext(full)[1], 'application/octet-stream'))
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()
        q = self.hub.subscribe()
        try:
            while True:
                try:
                    event, data = q.get(timeout=15)
                    chunk = 'event: %s\ndata: %s\n\n' % (event, data)
                except queue.Empty:
                    chunk = ': keep-alive\n\n'
                self.wfile.write(chunk.encode('utf-8'))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            self.hub.unsubscribe(q)


def main():
    parser = argparse.ArgumentParser(description='Platoon simulator dashboard')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--runs', default=DEFAULT_RUNS, help='folder with ia2c_catchup/ and ma2c_nc_catchup/')
    parser.add_argument('--open', action='store_true', help='open the dashboard in the default browser')
    parser.add_argument('--replay', nargs='+', metavar='LOG',
                        help='play back MPI run logs (mpi/logs/*.jsonl) instead of simulating')
    args = parser.parse_args()
    logging.disable(logging.INFO)

    hub = Hub()
    if args.replay:
        sim = Replay(hub, args.replay)
    else:
        print('Loading the trained policies ...', flush=True)
        sim = Simulation(hub, args.runs)
    threading.Thread(target=sim.run, daemon=True).start()
    Handler.sim, Handler.hub = sim, hub
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    url = 'http://localhost:%d' % args.port
    print('Platoon simulator running at %s   (Ctrl+C to stop)' % url, flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
