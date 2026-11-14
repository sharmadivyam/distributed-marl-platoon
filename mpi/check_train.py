"""
Proves that distributed training on MPI (mpi/train_mpi.py, each car learns in its own process)
does the same maths as the original central trainer (main.py train).

    python mpi/check_train.py [--steps 2400] [--seed 12]

Both start from the same initial weights and use the same random numbers (in the MPI run the road
process hands out the numbers the central trainer would draw). Then the test reward after every
episode and every weight of all 8 networks at the end must match.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'mpi', 'results', 'logs', 'check_train')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=2400)
    p.add_argument('--seed', type=int, default=12)
    args = p.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    mpi_dir, central_dir = os.path.join(OUT, 'mpi'), os.path.join(OUT, 'central')

    print('1/2 distributed training: 9 MPI processes ...', flush=True)
    subprocess.run(['mpirun', '--oversubscribe', '-np', '9', sys.executable, 'mpi/train_mpi.py',
                    '--seed', str(args.seed), '--steps', str(args.steps), '--mode', 'independent',
                    '--match-central', '--out', mpi_dir],
                   cwd=REPO, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('2/2 central training: main.py train, same config ...', flush=True)
    subprocess.run([sys.executable, 'main.py', '--base-dir', central_dir, 'train',
                    '--config-dir', os.path.join(mpi_dir, 'data', 'ia2c_catchup.ini')],
                   cwd=REPO, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    mpi_log = [json.loads(l) for l in open(os.path.join(mpi_dir, 'data', 'train_log.jsonl')) if '"step"' in l]
    central_log = pd.read_csv(os.path.join(central_dir, 'data', 'train_reward.csv'))
    print('\nepisode  steps  test reward: central      MPI')
    for k, (row, (_, c)) in enumerate(zip(mpi_log, central_log.iterrows())):
        print('%7d  %5d  %21.6f  %8.6f' % (k + 1, c['step'], c['avg_reward'], row['test_reward']))
    same_rewards = len(mpi_log) == len(central_log) and np.allclose(
        [r['test_reward'] for r in mpi_log], central_log['avg_reward'].values, rtol=0, atol=1e-6)

    def final(folder):
        ck = [f for f in os.listdir(folder) if f.startswith('checkpoint')]
        return torch.load(os.path.join(folder, ck[0]))['model_state_dict']
    a, b = final(os.path.join(central_dir, 'model')), final(os.path.join(mpi_dir, 'model'))
    assert a.keys() == b.keys()
    diff = max(float((a[k] - b[k]).abs().max()) for k in a)
    size = max(float(a[k].abs().max()) for k in a)
    print('\nfinal weights: %d tensors, largest weight %.3f, largest difference %.2e' % (len(a), size, diff))
    ok = same_rewards and diff < 1e-4
    print('RESULT:', 'IDENTICAL up to float32 rounding (distributed training = central training)' if ok
          else 'DIFFERENT')
    # the weight difference starts at ~3e-8 (one float32 rounding step at 0.5) after the first
    # updates and grows slowly with training length; any logic difference would change the rewards
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
