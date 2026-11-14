"""
Central training vs distributed training on MPI (IA2C, catch-up, 60k steps).
Central: seeds 12-14. MPI (each car alone / neighbours average): seeds 12-17.

    python mpi/compare_training.py

1. Evaluates every MPI-trained run that has no eva_data yet, with the original evaluator
   (main.py evaluate --checkpoint best) on the same 50 unseen seeds as the reproduction check.
2. Writes mpi/results/training_compare.csv and training_curves.png.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'demo'))
sys.path.insert(0, os.path.join(REPO, 'mpi'))
import make_report as mr  # noqa: E402
from experiments import OUT, plt, style, write_csv  # noqa: E402

SEEDS = (12, 13, 14, 15, 16, 17)
EVAL_SEEDS = ','.join(str(s) for s in range(100000, 100500, 10))
CENTRAL = ['demo/runs/ia2c_catchup', 'demo/repro/seed13/ia2c_catchup', 'demo/repro/seed14/ia2c_catchup']
METHODS = [  # (label, colour or None = table only, {seed: run folder})
    ('Central (original code)', '#52514e', dict(zip((12, 13, 14), CENTRAL))),
    ('MPI, each car alone', '#2a78d6', {s: 'mpi/train/independent_s%d' % s for s in SEEDS}),
    ('MPI, neighbours average', '#eb6834', {s: 'mpi/train/consensus_s%d' % s for s in SEEDS}),
    ('MPI, alone, local clipping', None, {s: 'mpi/train/localclip/independent_s%d' % s for s in (12, 13, 14)}),
    ('MPI, averaging, local clipping', None, {s: 'mpi/train/localclip/consensus_s%d' % s for s in (12, 13, 14)}),
]
FIXED_BASELINE = -72.7


def evaluate(run_dir):
    if os.path.isdir(os.path.join(REPO, run_dir, 'eva_data')):
        return
    with open(os.path.join(REPO, run_dir, 'eval_stdout.log'), 'w') as f:
        subprocess.run([sys.executable, 'main.py', '--base-dir', run_dir, 'evaluate',
                        '--evaluation-seeds', EVAL_SEEDS, '--checkpoint', 'best'],
                       cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=True)


def score(run_dir):
    import glob
    path = glob.glob(os.path.join(REPO, run_dir, 'eva_data', '*_traffic.csv'))
    return float(mr.episode_metrics(pd.read_csv(path[0]), 1, 20, 15, 8)['reward'].mean())


def curve(run_dir):
    """(steps, test reward after each training episode)"""
    log = os.path.join(REPO, run_dir, 'data', 'train_log.jsonl')
    if os.path.exists(log):
        rows = [json.loads(l) for l in open(log)]
        rows = [r for r in rows if 'step' in r]
        return np.array([r['step'] for r in rows]), np.array([r['test_reward'] for r in rows])
    df = pd.read_csv(os.path.join(REPO, run_dir, 'data', 'train_reward.csv'))
    return df['step'].values, df['avg_reward'].values


def minutes(run_dir):
    log = os.path.join(REPO, run_dir, 'data', 'train_log.jsonl')
    if os.path.exists(log):
        last = json.loads(open(log).read().strip().splitlines()[-1])
        return last['summary']['minutes'], last['summary']['consensus_mb_total']
    lines = [l for l in open(os.path.join(REPO, run_dir, 'train_stdout.log')) if l[:4].isdigit()]
    t0, t1 = (pd.Timestamp(l[:23].replace(',', '.')) for l in (lines[0], lines[-1]))
    return round((t1 - t0).total_seconds() / 60, 1), 0.0


def main():
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(evaluate, [d for _, _, dirs in METHODS[1:] for d in dirs.values()]))

    rows = []
    for label, _, dirs in METHODS:
        scores = {s: score(d) for s, d in dirs.items()}
        times = [minutes(d) for d in dirs.values()]
        v = list(scores.values())
        rows.append({'method': label, **{'seed%d' % s: round(scores[s], 1) if s in scores else '' for s in SEEDS},
                     'mean': round(np.mean(v), 1), 'median': round(np.median(v), 1),
                     'seeds_beating_fixed': '%d/%d' % (sum(x > FIXED_BASELINE for x in v), len(v)),
                     'train_minutes': round(np.mean([t for t, _ in times]), 1),
                     'weights_sent_mb_per_run': round(np.mean([m for _, m in times]), 1)})
    write_csv(rows, os.path.join(OUT, 'training_compare.csv'))
    for r in rows:
        print({k: (float(x) if isinstance(x, np.floating) else x) for k, x in r.items()})

    # learning curves: rolling mean of the last 5 test episodes (what picks the best checkpoint)
    grid = np.linspace(3000, 60000, 120)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for label, colour, dirs in METHODS:
        if colour is None:
            continue
        ys = []
        for d in dirs.values():
            x, y = curve(d)
            y = pd.Series(y).rolling(5, min_periods=1).mean().values
            ys.append(np.interp(grid, x, y))
        ys = np.array(ys)
        ax.plot(grid / 1000, ys.mean(0), color=colour, lw=2, label=label)
        ax.fill_between(grid / 1000, ys.min(0), ys.max(0), color=colour, alpha=0.12, lw=0)
    ax.axhline(FIXED_BASELINE, color='#a3a8ae', ls='--', lw=1.2, label='Fixed controller')
    ax.set_ylim(-700, 0)
    style(ax, 'training steps (thousands)', 'test reward (rolling mean of 5)',
          'Learning curves (line = mean over seeds, band = best to worst seed)')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'training_curves.png'), dpi=160)
    print('wrote', os.path.join(OUT, 'training_curves.png'))


if __name__ == '__main__':
    main()
