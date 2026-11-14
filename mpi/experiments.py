"""
Experiments on the MPI platoon, saved to mpi/results/ (CSV + plots):

  1. message loss sweep:  0..50 % lost V2V messages
  2. delay sweep:         0..1 s late V2V messages
     (both for the fixed controller and IA2C, each with the two fallbacks)
  3. scaling:             8, 16, 32 cars (fixed controller), time per step and V2V traffic

    python mpi/experiments.py              # everything (about 20 minutes)
    python mpi/experiments.py --only scaling
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'mpi', 'results')
LOGS = os.path.join(OUT, 'logs')
PROG = os.path.join(REPO, 'mpi', 'platoon_mpi.py')
COLORS = {('fixed', 'last'): '#52514e', ('fixed', 'radar'): '#a3a8ae',
          ('ia2c', 'last'): '#eb6834', ('ia2c', 'radar'): '#f5b08f'}
LABELS = {'fixed': 'Fixed controller', 'ia2c': 'IA2C'}


def run(controller, cars=8, timeout_ms=30, **flags):
    name = '%s_n%d_%s' % (controller, cars, '_'.join('%s%s' % (k, v) for k, v in sorted(flags.items())) or 'clean')
    log = os.path.join(LOGS, name + '.jsonl')
    cmd = ['mpirun', '--oversubscribe', '-np', str(cars + 1), sys.executable, PROG, '--controller', controller,
           '--timeout-ms', str(timeout_ms), '--log', log]
    if cars != 8:
        cmd += ['--cars', str(cars)]
    for key, value in flags.items():
        cmd += ['--' + key.replace('_', '-'), str(value)]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)
    with open(log) as f:
        summary = [json.loads(line)['summary'] for line in f if line.startswith('{"summary"')][0]
    summary.pop('rewards')
    summary['log'] = os.path.relpath(log, REPO)
    print('  %-55s avg reward %8.1f  collision %-5s  %6.1f ms/step'
          % (name, summary['avg_reward'], summary['collision_t'], summary['ms_per_step']), flush=True)
    return summary


def write_csv(rows, path):
    keys = [k for k in rows[0] if not isinstance(rows[0][k], dict)]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontsize=11)
    ax.grid(True, color='#e4e3df')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def sweep(kind, values, fmt):
    rows = []
    for controller in ('fixed', 'ia2c'):
        for fallback in ('last', 'radar'):
            for value in values:
                flags = {'fallback': fallback}
                if value:
                    flags[kind] = value
                s = run(controller, **flags)
                s.update({'sweep': kind, 'value': value})
                rows.append(s)
    write_csv(rows, os.path.join(OUT, '%s_sweep.csv' % kind))
    plot_sweep(kind, rows, fmt)
    return rows


def plot_sweep(kind, rows, fmt):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for controller in ('fixed', 'ia2c'):
        for fallback in ('last', 'radar'):
            pts = [r for r in rows if r['controller'] == controller and r['fallback'] == fallback]
            xs = [fmt(float(r['value'])) for r in pts]
            ys = [float(r['avg_reward']) for r in pts]
            ax.plot(xs, ys, marker='o', color=COLORS[(controller, fallback)], linewidth=2,
                    label='%s, fallback: %s' % (LABELS[controller], 'radar estimate' if fallback == 'radar' else 'last value'))
            for r, x, y in zip(pts, xs, ys):
                if r['collision_t'] not in (None, '', 'None'):     # runs that ended in a crash
                    ax.scatter([x], [y], s=150, facecolors='none', edgecolors='#e5484d', linewidths=2, zorder=5)
    if kind == 'delay':                       # rewards span -23 to -470: log scale, readable ticks
        ax.set_yscale('symlog', linthresh=10)
        ticks = [-20, -30, -50, -100, -200, -500]
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.set_ylim(-600, -15)
    style(ax, 'V2V messages lost [%]' if kind == 'drop' else 'V2V message delay [s]',
          'avg reward per step (higher is better)',
          'Platoon under %s (red ring = collision)' % ('message loss' if kind == 'drop' else 'message delay'))
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '%s_sweep.png' % kind), dpi=160)
    plt.close(fig)


def scaling():
    rows = []
    for cars in (8, 16, 32):
        # 30 s runs (before long platoons start to collide), 3 repeats, keep the fastest (timing noise)
        reps = [run('fixed', cars=cars, timeout_ms=5000, steps=300) for _ in range(3)]
        s = min(reps, key=lambda r: r['ms_per_step'])
        s['ms_per_step_all'] = ' '.join('%.2f' % r['ms_per_step'] for r in reps)
        s['msgs_per_step'] = s['msgs_sent'] / s['steps']
        rows.append(s)
    write_csv(rows, os.path.join(OUT, 'scaling.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    cars = [r['cars'] for r in rows]
    axes[0].plot(cars, [r['ms_per_step'] for r in rows], marker='o', color='#2a78d6', linewidth=2)
    style(axes[0], 'cars (MPI processes = cars + 1)', 'wall time per step [ms]', 'Time per 0.1 s step')
    axes[1].plot(cars, [r['msgs_per_step'] for r in rows], marker='o', color='#eb6834', linewidth=2)
    style(axes[1], 'cars (MPI processes = cars + 1)', 'V2V messages per step', 'Communication grows linearly')
    for ax in axes:
        ax.set_xticks(cars)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'scaling.png'), dpi=160)
    plt.close(fig)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', choices=['drop', 'delay', 'scaling'], default=None)
    p.add_argument('--plot-only', action='store_true', help='redraw the sweep graphs from the saved CSVs')
    args = p.parse_args()
    if args.plot_only:
        for kind, fmt in (('drop', lambda v: 100 * v), ('delay', lambda v: v * 0.1)):
            with open(os.path.join(OUT, '%s_sweep.csv' % kind)) as f:
                plot_sweep(kind, list(csv.DictReader(f)), fmt)
        print('graphs redrawn in', os.path.relpath(OUT, REPO))
        return
    os.makedirs(LOGS, exist_ok=True)
    if args.only in (None, 'drop'):
        print('1) message loss sweep')
        sweep('drop', [0, 0.1, 0.2, 0.3, 0.5], lambda v: 100 * v)
    if args.only in (None, 'delay'):
        print('2) delay sweep')
        sweep('delay', [0, 1, 2, 3, 5, 10], lambda v: v * 0.1)
    if args.only in (None, 'scaling'):
        print('3) scaling')
        scaling()
    print('results in', os.path.relpath(OUT, REPO))


if __name__ == '__main__':
    main()
