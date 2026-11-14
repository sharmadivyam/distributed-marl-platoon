"""
Turn demo/runs/* into plots + a results table in demo/results/.

Usage (from repo root, after demo/run_demo.py):
    python demo/make_report.py

Outputs:
    results/learning_curves_<scenario>.png   reward vs training step, per algorithm
    results/trajectories_<scenario>.png      headway + speed of all 8 cars in one test episode
    results/summary.md                       evaluation table (50 unseen seeds)

Baseline = the same OVM car-following controller with a FIXED preset (no learning),
evaluated on the same seeds. The best of the 4 presets is shown.
"""
import argparse
import configparser
import glob
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from envs.cacc_env import CACCEnv  # noqa: E402

LABEL = {'ma2c_nc': 'NeurComm', 'ia2c': 'IA2C', 'ma2c_cnet': 'CommNet', 'ma2c_dial': 'DIAL',
         'ia2c_fp': 'IA2C-FP', 'ma2c_cu': 'IA2C-CU'}
# fixed colour per algorithm (never by rank), validated categorical slots
COLOR = {'ma2c_nc': '#2a78d6', 'ia2c': '#eb6834', 'ma2c_cnet': '#1baf7a', 'ma2c_dial': '#eda100',
         'ia2c_fp': '#e87ba4', 'ma2c_cu': '#4a3aa7'}
BASELINE_COLOR = '#52514e'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
DEFAULT_SEEDS = ','.join(str(s) for s in range(100000, 100500, 10))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default=os.path.join(REPO, 'demo', 'runs'))
    p.add_argument('--out', default=os.path.join(REPO, 'demo', 'results'))
    p.add_argument('--evaluation-seeds', default=DEFAULT_SEEDS,
                   help='must match the seeds used by run_demo.py')
    return p.parse_args()


def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def find_runs(runs_dir):
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, '*'))):
        ini = glob.glob(os.path.join(run_dir, 'data', '*.ini'))
        if not ini:
            continue
        config = configparser.ConfigParser()
        config.read(ini[0])
        env_cfg = config['ENV_CONFIG']
        runs.append({'dir': run_dir, 'agent': env_cfg.get('agent'),
                     'scenario': env_cfg.get('scenario').split('_')[1], 'config': config})
    return runs


def episode_metrics(traffic, h_min, h_star, v_star, n_agent):
    """Per-episode metrics from an eva_data/*_traffic.csv frame (or the same layout)."""
    rows = []
    for ep, df in traffic.groupby('episode'):
        df = df[df['time_sec'] > 0]  # first row is the initial state (reward 0)
        hs = df[['headway_%d_m' % (i + 1) for i in range(n_agent)]].values
        vs = df[['velocity_%d_mps' % (i + 1) for i in range(n_agent)]].values
        rows.append({'episode': ep,
                     'reward': df['reward'].mean(),
                     'collision': bool(hs.min() < h_min),
                     'headway_err': np.abs(hs - h_star).mean(),
                     'speed_err': np.abs(vs - v_star).mean()})
    return pd.DataFrame(rows)


def evaluated_checkpoint(run_dir):
    """('best'|'last', step) of the checkpoint the evaluation loaded, read from eva_log."""
    found = None
    for log in sorted(glob.glob(os.path.join(run_dir, 'eva_log', '*.log'))):
        for line in open(log):
            if 'Checkpoint loaded:' in line:
                path = line.strip().split('Checkpoint loaded: ')[-1]
                kind = 'best' if '/best/' in path else 'last'
                found = (kind, int(path.split('checkpoint-')[-1].split('.')[0]))
    return found


def run_fixed_preset(env_cfg, seeds, preset):
    """Roll out the OVM controller with one fixed preset; return a traffic frame like eva_data."""
    env = CACCEnv(env_cfg)
    env.train_mode = False
    env.init_test_seeds(seeds)
    env.init_data(True, False, None)
    for k in range(len(seeds)):
        env.reset(test_ind=k)
        while True:
            _, _, done, _ = env.step([preset] * env.n_agent)
            if done:
                break
    return pd.concat(env.traffic_data), env


def plot_learning_curves(scen_runs, baseline_reward, scenario, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    curves = []
    for run in scen_runs:
        df = pd.read_csv(os.path.join(run['dir'], 'data', 'train_reward.csv'))
        # rolling median: robust to single crash episodes (reward ~ -8000)
        smooth = df['avg_reward'].rolling(15, min_periods=1).median()
        ax.plot(df['step'], smooth, color=COLOR.get(run['agent'], INK), linewidth=2,
                label=LABEL.get(run['agent'], run['agent']))
        curves.append(smooth.values)
    ax.axhline(baseline_reward, color=BASELINE_COLOR, linestyle='--', linewidth=1.5,
               label='Best fixed controller (no learning)')
    # keep the interesting range visible: crashes early in training would squash it
    lo = np.percentile(np.concatenate(curves), 5) if curves else baseline_reward
    ax.set_ylim(min(lo, baseline_reward) * 1.15, 0)
    ax.set_xlabel('Training step', color=INK)
    ax.set_ylabel('Avg reward per step (higher is better)', color=INK)
    ax.set_title('CACC %s: learning curves (test episode after each training episode)' % scenario,
                 color=INK, fontsize=11, loc='left')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: '%gk' % (x / 1000)))
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_trajectories(panels, env, scenario, out_path):
    """panels: list of (title, traffic frame for one episode)."""
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 5.6), sharex=True, sharey='row', squeeze=False)
    shades = plt.cm.Blues(np.linspace(0.4, 1.0, env.n_agent))
    for c, (title, df) in enumerate(panels):
        t = df['time_sec'].values
        for i in range(env.n_agent):
            axes[0, c].plot(t, df['headway_%d_m' % (i + 1)], color=shades[i], linewidth=1.3,
                            label='car %d' % (i + 1))
            axes[1, c].plot(t, df['velocity_%d_mps' % (i + 1)], color=shades[i], linewidth=1.3)
        axes[0, c].axhline(env.h_star, color=MUTED, linestyle=':', linewidth=1)
        axes[1, c].axhline(env.v_star, color=MUTED, linestyle=':', linewidth=1)
        axes[0, c].set_title(title, color=INK, fontsize=10, loc='left')
        axes[1, c].set_xlabel('Time [s]', color=INK)
        for r in range(2):
            style(axes[r, c])
    axes[0, 0].set_ylabel('Headway [m]  (target %.0f)' % env.h_star, color=INK)
    axes[1, 0].set_ylabel('Speed [m/s]  (target %.0f)' % env.v_star, color=INK)
    axes[0, -1].legend(frameon=False, fontsize=7, loc='upper right', ncol=2)
    fig.suptitle('CACC %s: one unseen test episode (seed %d), car 1 = front of platoon'
                 % (scenario, env.test_seeds[0]), color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fmt(m):
    return ('%.1f ± %.1f' % (m['reward'].mean(), m['reward'].std()),
            '%d / %d' % (m['collision'].sum(), len(m)),
            '%.2f' % m['headway_err'].mean(), '%.2f' % m['speed_err'].mean())


def main():
    args = parse_args()
    logging.disable(logging.INFO)  # env logs its action map on every construction
    seeds = [int(s) for s in args.evaluation_seeds.split(',')]
    os.makedirs(args.out, exist_ok=True)
    runs = find_runs(args.runs)
    if not runs:
        sys.exit('No runs found in %s. Run demo/run_demo.py first.' % args.runs)

    lines = ['# Demo results', '',
             'Evaluation: %d unseen test seeds (%d–%d), deterministic policy. '
             'Reward is the average per-step platoon reward (higher is better, 0 is perfect). '
             'Errors are mean absolute deviation from target headway / speed over all cars and steps. '
             '"best checkpoint" = the training checkpoint with the highest rolling mean of 5 validation '
             'episodes; validation seeds count up from 11 and stay far below the evaluation seeds.'
             % (len(seeds), seeds[0], seeds[-1]), '']
    for scenario in sorted({r['scenario'] for r in runs}):
        scen_runs = [r for r in runs if r['scenario'] == scenario]
        env_cfg = scen_runs[0]['config']['ENV_CONFIG']

        # no-learning baseline: best fixed preset of the same controller
        best = None
        for preset in range(4):
            traffic, env = run_fixed_preset(env_cfg, seeds, preset)
            m = episode_metrics(traffic, env.h_min, env.h_star, env.v_star, env.n_agent)
            if best is None or m['reward'].mean() > best[1]['reward'].mean():
                best = (preset, m, traffic, env)
        preset, base_m, base_traffic, env = best

        rows = [('Fixed controller, preset %s (no learning)' % (env.a_map[preset],),) + fmt(base_m)]
        panels = [('Fixed controller %s' % (env.a_map[preset],),
                   base_traffic[base_traffic['episode'] == 1])]
        for run in scen_runs:
            path = glob.glob(os.path.join(run['dir'], 'eva_data', '*_traffic.csv'))
            if not path:
                print('warning: %s has no evaluation data, skipping' % run['dir'])
                continue
            traffic = pd.read_csv(path[0])
            m = episode_metrics(traffic, env.h_min, env.h_star, env.v_star, env.n_agent)
            steps = pd.read_csv(os.path.join(run['dir'], 'data', 'train_reward.csv'))['step'].max()
            name = LABEL.get(run['agent'], run['agent'])
            ckpt = evaluated_checkpoint(run['dir'])
            if ckpt is None:
                desc = '%gk training steps' % round(steps / 1000)
            else:
                desc = '%s checkpoint @ %gk of %gk steps' % (ckpt[0], round(ckpt[1] / 1000, 1),
                                                             round(steps / 1000))
            rows.append(('%s (%s)' % (name, desc),) + fmt(m))
            panels.append((name, traffic[traffic['episode'] == traffic['episode'].min()]))

        plot_learning_curves(scen_runs, base_m['reward'].mean(), scenario,
                             os.path.join(args.out, 'learning_curves_%s.png' % scenario))
        plot_trajectories(panels, env, scenario, os.path.join(args.out, 'trajectories_%s.png' % scenario))

        lines += ['## CACC %s' % scenario, '',
                  '| Method | Reward (mean ± std) | Collisions | Headway err [m] | Speed err [m/s] |',
                  '|---|---|---|---|---|']
        lines += ['| %s | %s | %s | %s | %s |' % r for r in rows]
        lines += ['', '![learning curves](learning_curves_%s.png)' % scenario, '',
                  '![trajectories](trajectories_%s.png)' % scenario, '']

    summary = os.path.join(args.out, 'summary.md')
    with open(summary, 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print('\nWrote %s and plots to %s' % (os.path.relpath(summary, REPO), os.path.relpath(args.out, REPO)))


if __name__ == '__main__':
    main()
