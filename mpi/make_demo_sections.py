"""
Records the MPI runs for the demo page and writes mpi/demo_sections.json, which groups them into
sections (one situation each, all controllers side by side).

    python mpi/make_demo_sections.py
    python -m simulator.server --port 8767 --open --sections mpi/demo_sections.json
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROG = os.path.join('mpi', 'platoon_mpi.py')
LOGS = os.path.join('mpi', 'logs', 'sections')
ALL = ['radar', 'fixed', 'ia2c', 'neurcomm']
CONTROLLERS = ['fixed', 'ia2c', 'neurcomm']

SECTIONS = [
    ('normal', 'Normal driving, no faults', ALL, [],
     'Catch-up: car 1 starts 36 m behind the leader and the platoon has to close up smoothly. '
     'With only their own radar and no messages, the cars crash at 11 s.'),
    ('brake', 'Leader brakes at 30 s', ALL, ['--brake-at', '30'],
     'The leader slows from 15 to 8 m/s in about 5 s and speeds up again. Without messages the platoon '
     'has already crashed at 11 s.'),
    ('delay', 'Messages arrive 0.5 s late', CONTROLLERS, ['--delay', '5', '--fallback', 'last', '--fast-faults'],
     'Every V2V message arrives 5 steps late, and each car uses the newest (old) value it has.'),
    ('delay_radar', 'Messages 0.5 s late, radar fallback', CONTROLLERS,
     ['--delay', '5', '--fallback', 'radar', '--fast-faults'],
     'Same delay, but a car estimates the front car\'s speed from its own radar instead of trusting '
     'an old message. NeurComm\'s other message parts (policy, hidden state) cannot be estimated.'),
    ('drop', 'Half of all messages lost', CONTROLLERS, ['--drop', '0.5', '--fallback', 'last', '--fast-faults'],
     'Each V2V message is lost with probability 50%; a car reuses the last value it received.'),
    ('crash', 'Car 4 crashes at 25 s, then the leader brakes', CONTROLLERS,
     ['--kill-car', '4', '--kill-at', '25', '--brake-at', '30', '--crash-timeout-ms', '500', '--timeout-ms', '50'],
     'Car 4\'s process goes silent (crash-stop). The road detects it with a timeout and the car rolls '
     'on at constant speed, so it cannot brake when the leader does.'),
    ('liar', 'Car 4 is malicious, no defence', CONTROLLERS,
     ['--byzantine-car', '4', '--byzantine-at', '25', '--attack', 'noise', '--brake-at', '30'],
     'From 25 s car 4 drives normally but sends random values (Byzantine fault: it keeps talking, but '
     'lies). The leader brakes at 30 s and the cars behind car 4 believe its messages.'),
    ('liar_defense', 'Car 4 is malicious, radar check on', CONTROLLERS,
     ['--byzantine-car', '4', '--byzantine-at', '25', '--attack', 'noise', '--brake-at', '30', '--defense'],
     'Same attack. Every car compares its front car\'s claimed speed with what its own radar says '
     '(how fast the gap changes) and ignores messages that disagree by more than 2 m/s.'),
]


def record(job):
    key, controller, flags = job
    log = os.path.join(LOGS, '%s_%s.jsonl' % (key, controller))
    if '--timeout-ms' not in flags:     # wait long enough that a busy laptop never looks like a fault
        flags = ['--timeout-ms', '5000'] + flags
    cmd = ['mpirun', '--oversubscribe', '-np', '9', sys.executable, PROG, '--controller', controller,
           '--log', log] + flags
    full = os.path.join(REPO, log)
    if not (os.path.exists(full) and '"summary"' in open(full).read()[-300000:]):
        subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)
    return log


def main():
    jobs = [(key, c, flags) for key, _, controllers, flags, _ in SECTIONS for c in controllers]
    with ThreadPoolExecutor(3) as pool:
        logs = dict(zip([(k, c) for k, c, _ in jobs], pool.map(record, jobs)))
    out = [{'title': title, 'about': about, 'logs': [logs[(key, c)] for c in controllers]}
           for key, title, controllers, _, about in SECTIONS]
    path = os.path.join(REPO, 'mpi', 'demo_sections.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('recorded %d runs in %s, sections in mpi/demo_sections.json' % (len(jobs), LOGS))


if __name__ == '__main__':
    main()
