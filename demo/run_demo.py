"""
Train + evaluate several algorithms on CACC in parallel, for a demo.

Usage (from repo root):
    python demo/run_demo.py                      # defaults below
    python demo/run_demo.py --steps 2e4          # quick smoke run (~5-10 min)
    python demo/run_demo.py --algos ma2c_nc,ia2c --scenarios catchup,slowdown

Each run lives in demo/runs/<algo>_<scenario>/ with the usual main.py layout
(data/, log/, model/, eva_data/). Runs that already have a checkpoint are not retrained.
"""
import argparse
import configparser
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# agent string -> prefix of the config file in config/
CONFIG_PREFIX = {'ia2c': 'ia2c', 'ia2c_fp': 'ia2c_fp', 'ma2c_cu': 'ia2c_cu',
                 'ma2c_nc': 'ma2c_nc', 'ma2c_cnet': 'ma2c_cnet', 'ma2c_dial': 'ma2c_dial'}
# 50 test seeds far above the seeds training reaches (it starts at 12 and adds 2 per episode)
DEFAULT_SEEDS = ','.join(str(s) for s in range(100000, 100500, 10))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=float, default=2.5e5, help='training steps per run')
    p.add_argument('--algos', default='ma2c_nc,ia2c,ma2c_cnet')
    p.add_argument('--scenarios', default='catchup')
    p.add_argument('--out', default=os.path.join(REPO, 'demo', 'runs'))
    p.add_argument('--evaluation-seeds', default=DEFAULT_SEEDS)
    p.add_argument('--optimizer', choices=['rmsprop', 'rmsprop_tf'], default='rmsprop',
                   help='rmsprop: torch.optim.RMSprop (original code); rmsprop_tf: TF1-style, steadier but not better in our tests')
    p.add_argument('--checkpoint', choices=['best', 'last'], default='best',
                   help='which checkpoint to evaluate (best = best rolling test reward in training)')
    p.add_argument('--seed', type=int, default=None, help='training seed (default: the config value, 12)')
    p.add_argument('--coop-gamma', type=float, default=None,
                   help='spatial discount alpha (default: the config value; -1 means one global reward)')
    return p.parse_args()


def write_config(algo, scenario, steps, optimizer, cfg_dir, seed=None, coop_gamma=None):
    src = os.path.join(REPO, 'config', 'config_%s_%s.ini' % (CONFIG_PREFIX[algo], scenario))
    config = configparser.ConfigParser()
    config.read(src)
    # only the training length and optimizer implementation change; hyperparameters are the paper's
    config['TRAIN_CONFIG']['total_step'] = str(int(steps))
    config['MODEL_CONFIG']['optimizer'] = optimizer
    if seed is not None:
        config['ENV_CONFIG']['seed'] = str(seed)
    if coop_gamma is not None:
        config['ENV_CONFIG']['coop_gamma'] = str(coop_gamma)
    dst = os.path.join(cfg_dir, '%s_%s.ini' % (algo, scenario))
    with open(dst, 'w') as f:
        config.write(f)
    return dst


def has_checkpoint(run_dir):
    model_dir = os.path.join(run_dir, 'model')
    return os.path.isdir(model_dir) and any(f.startswith('checkpoint') for f in os.listdir(model_dir))


def run_parallel(jobs, label):
    """jobs: list of (name, cmd, log_path). Runs all at once, waits, reports."""
    procs = []
    for name, cmd, log_path in jobs:
        log = open(log_path, 'w')
        procs.append((name, subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT), log))
        print('[%s] started %s  (log: %s)' % (label, name, os.path.relpath(log_path, REPO)))
    t0 = time.time()
    failed = []
    for name, proc, log in procs:
        rc = proc.wait()
        log.close()
        print('[%s] %s finished with code %d after %.1f min' % (label, name, rc, (time.time() - t0) / 60))
        if rc != 0:
            failed.append(name)
    return failed


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg_dir = os.path.join(args.out, 'configs')
    os.makedirs(cfg_dir, exist_ok=True)

    runs = []
    for scenario in args.scenarios.split(','):
        for algo in args.algos.split(','):
            name = '%s_%s' % (algo, scenario)
            runs.append((name, os.path.join(args.out, name), write_config(algo, scenario, args.steps, args.optimizer, cfg_dir, args.seed, args.coop_gamma)))

    train_jobs = []
    for name, run_dir, cfg in runs:
        if has_checkpoint(run_dir):
            print('[train] %s already has a checkpoint, skipping (delete %s to retrain)' %
                  (name, os.path.relpath(run_dir, REPO)))
            continue
        os.makedirs(run_dir, exist_ok=True)
        cmd = [sys.executable, 'main.py', '--base-dir', run_dir, 'train', '--config-dir', cfg]
        train_jobs.append((name, cmd, os.path.join(run_dir, 'train_stdout.log')))
    failed = run_parallel(train_jobs, 'train') if train_jobs else []

    eval_jobs = []
    for name, run_dir, _ in runs:
        if name in failed:
            continue
        cmd = [sys.executable, 'main.py', '--base-dir', run_dir, 'evaluate',
               '--evaluation-seeds', args.evaluation_seeds, '--checkpoint', args.checkpoint]
        eval_jobs.append((name, cmd, os.path.join(run_dir, 'eval_stdout.log')))
    failed += run_parallel(eval_jobs, 'eval')

    if failed:
        print('FAILED: %s (see the logs above)' % ', '.join(failed))
        sys.exit(1)
    print('All runs done. Next: python demo/make_report.py')


if __name__ == '__main__':
    main()
