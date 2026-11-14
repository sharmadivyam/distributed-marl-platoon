"""
Checks that the MPI platoon gives exactly the same result as the Python simulator (and therefore
as the report's evaluation): same reward at every one of the 600 steps.

    python mpi/check_mpi.py
"""
import json
import logging
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
logging.disable(logging.INFO)

from simulator.check import lane_rollout  # noqa: E402
from simulator.controllers import build_lane  # noqa: E402

SEED = 100000
LOG_DIR = os.path.join(REPO, 'mpi', 'logs')


def mpi_rewards(controller):
    log = os.path.join(LOG_DIR, 'check_%s.jsonl' % controller)
    cmd = ['mpirun', '--oversubscribe', '-np', '9', sys.executable, os.path.join(REPO, 'mpi', 'platoon_mpi.py'),
           '--controller', controller, '--seed', str(SEED), '--timeout-ms', '5000', '--log', log]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)
    with open(log) as f:
        summary = [json.loads(line)['summary'] for line in f if line.startswith('{"summary"')][0]
    return np.array(summary['rewards']), summary


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    ok = True
    for controller in ('fixed', 'ia2c'):
        ref, _ = lane_rollout(build_lane(controller), 600, seed=SEED)
        got, s = mpi_rewards(controller)
        same = len(ref) == len(got) and np.array_equal(ref, got)
        ok &= same
        print('%-6s %s  python %.2f  mpi %.2f  (%d processes, %d V2V messages)'
              % (controller, 'IDENTICAL at all %d steps' % len(ref) if same else 'DIFFERENT',
                 ref.mean(), got.mean(), s['processes'], s['msgs_sent']))
    print('->', 'MPI matches Python exactly' if ok else 'MISMATCH')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
