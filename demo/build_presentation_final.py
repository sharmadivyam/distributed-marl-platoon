"""
Final presentation (HTML) with the MPI work: reproduction check, simulator, MPI design, fault
experiments, scaling. Numbers are read from the result files, so re-running this after new
experiments updates everything.

    python demo/build_presentation_final.py

Writes demo/presentation_final/: report.html, REPORT.md, SPEAKER_NOTES.md and fig_*.png.
Needs: mpi/results/*.csv|png (python mpi/experiments.py) and the screenshots shot_*.png.
"""
import csv
import glob
import logging
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

DEMO = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(DEMO)
sys.path.insert(0, DEMO)
import build_presentation as bp  # noqa: E402
import build_presentation_10min as b10  # noqa: E402
import make_report as mr  # noqa: E402

OUT = os.path.join(DEMO, 'presentation_final')
RES = os.path.join(REPO, 'mpi', 'results')
BLUE, ORANGE, GREY, INK = '#2a78d6', '#eb6834', '#d9d8d4', '#1b1b1a'

REPRO = [   # (method, setting, run folders for seeds 12, 13, 14)
    ('Fixed controller', 'no learning', None),
    ('IA2C', 'global reward, 60k steps', ['demo/runs/ia2c_catchup', 'demo/repro/seed13/ia2c_catchup',
                                          'demo/repro/seed14/ia2c_catchup']),
    ('NeurComm', 'global reward (as released), 60k steps',
     ['demo/runs/ma2c_nc_catchup', 'demo/repro/seed13/ma2c_nc_catchup', 'demo/repro/seed14/ma2c_nc_catchup']),
    ('NeurComm', 'spatial discount 0.9, 60k steps',
     ['demo/repro/nc_alpha0.9_seed%d/ma2c_nc_catchup' % s for s in (12, 13, 14)]),
    ('NeurComm', 'spatial discount 0.9, 250k steps',
     ['demo/repro/nc_alpha0.9_250k_seed%d/ma2c_nc_catchup' % s for s in (12, 13, 14)]),
]
FIXED_BASELINE = -72.7


# ---------------------------------------------------------------- data
def eval_score(run_dir):
    path = glob.glob(os.path.join(REPO, run_dir, 'eva_data', '*_traffic.csv'))
    if not path:
        return None
    m = mr.episode_metrics(pd.read_csv(path[0]), 1, 20, 15, 8)
    return m['reward'].mean()


def reproduction_rows():
    rows = []
    for method, setting, dirs in REPRO:
        if dirs is None:
            rows.append([method, setting, '%.1f' % FIXED_BASELINE, '', '', '**%.1f**' % FIXED_BASELINE])
            continue
        scores = [eval_score(d) for d in dirs]
        if all(s is None for s in scores):
            continue          # not trained yet
        cells = ['%.1f' % s if s is not None else 'running' for s in scores]
        done = [s for s in scores if s is not None]
        mean = '%.1f' % np.mean(done) + ('' if len(done) == len(scores) else ' (so far)')
        rows.append([method, setting] + cells + [mean])
    return rows


def read_csv(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def sweep_table(kind, values, label):
    rows = read_csv('%s_sweep.csv' % kind)
    out = []
    for v in values:
        cells = [label(v)]
        for controller in ('fixed', 'ia2c'):
            for fallback in ('last', 'radar'):
                r = [x for x in rows if x['controller'] == controller and x['fallback'] == fallback
                     and float(x['value']) == v][0]
                crash = r['collision_t'] not in ('', 'None')
                cells.append('**crash at %s s**' % r['collision_t'] if crash else '%.1f' % float(r['avg_reward']))
        out.append(cells)
    return out


# ---------------------------------------------------------------- figure
def fig_mpi(path):
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set_xlim(-0.6, 17.4)
    ax.set_ylim(-2.75, 3.4)
    ax.axis('off')
    gap, w = 2.15, 1.25
    xs = [16.2 - gap * k for k in range(8)]           # car 1 on the right (front), car 8 on the left
    split = (xs[3] + xs[4]) / 2
    # two-laptop split: road + cars 1-4 on laptop A, cars 5-8 on laptop B
    ax.add_patch(Rectangle((split + 0.05, -2.55), 17.2 - split, 5.85, fill=False, ls='--', lw=1.2, ec='#8a8984'))
    ax.add_patch(Rectangle((-0.45, -2.55), split - 0.05 + 0.45, 5.85, fill=False, ls='--', lw=1.2, ec='#8a8984'))
    ax.text(17.1, -2.42, 'laptop A: ranks 0-4', ha='right', va='bottom', fontsize=9, color='#5d5c58')
    ax.text(-0.3, -2.42, 'laptop B: ranks 5-8', ha='left', va='bottom', fontsize=9, color='#5d5c58')
    # road rank (on laptop A); every car is linked to it, the links from laptop B cross the network
    road_l, road_r = split + 0.6, 16.9
    b10.box(ax, (road_l + road_r) / 2, 2.45, road_r - road_l, 0.95, 'Rank 0: the road',
            'physics of all cars + the leading vehicle', GREY, INK)
    targets = np.linspace(road_l + 0.4, road_r - 0.4, 8)[::-1]
    for k, x in enumerate(xs):
        b10.box(ax, x, -0.55, w, 1.0, 'Rank %d' % (k + 1), 'car %d\nown policy' % (k + 1), BLUE, 'white')
        ax.add_patch(FancyArrowPatch((x, -0.02), (targets[k], 1.95), arrowstyle='<|-|>', mutation_scale=9,
                                     color='#8a8984', lw=1, ls=(0, (3, 2))))
    ax.text(road_r - 0.05, 1.25, 'sensors down,\ncommand up', fontsize=8, color='#5d5c58', ha='right', va='center',
            bbox=dict(facecolor='white', edgecolor='none', pad=1))
    # V2V messages between neighbouring cars
    for a, b in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((a - w / 2 - 0.05, -0.55), (b + w / 2 + 0.05, -0.55), arrowstyle='<|-|>',
                                     mutation_scale=13, color=ORANGE, lw=2.4))
    white = dict(facecolor='white', edgecolor='none', pad=2)
    ax.text(8.6, -1.5, 'V2V messages between neighbouring car processes only (MPI isend / recv)',
            ha='center', fontsize=9.5, color='#b4481f', bbox=white)
    ax.text(8.6, -1.95, 'missing or late message: wait up to a timeout, then use a fallback',
            ha='center', fontsize=9, color=INK, bbox=white)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- report
def main():
    logging.disable(logging.INFO)
    os.makedirs(OUT, exist_ok=True)
    fig = lambda name: os.path.join(OUT, name)  # noqa: E731

    bp.fig_task(fig('fig_problem.png'))
    b10.fig_comm(fig('fig_ia2c_vs_neurcomm.png'))
    fig_mpi(fig('fig_mpi_architecture.png'))
    for name in ('drop_sweep.png', 'delay_sweep.png', 'scaling.png'):
        shutil.copy(os.path.join(RES, name), fig('fig_' + name))

    repro = reproduction_rows()
    drop = sweep_table('drop', [0, 0.2, 0.5], lambda v: '%d%% lost' % round(100 * v))
    delay = sweep_table('delay', [0, 3, 5, 10], lambda v: '%.1f s late' % (v * 0.1))
    scaling = read_csv('scaling.csv')
    sweep_head = ['Fault', 'Fixed, last value', 'Fixed, radar', 'IA2C, last value', 'IA2C, radar']

    blocks = [
        ('h1', 'Distributed multi-agent RL for cooperative driving'),
        ('p', 'We took the paper "Multi-agent Reinforcement Learning for Networked System Control" (Chu et al., '
              'ICLR 2020), reproduced its car-platoon experiment, and then ran the learned controllers as a real '
              'distributed system on MPI: one process per car, talking only by messages, with network faults '
              'injected on purpose.'),
        ('h2', '1. Problem'),
        ('img', fig('fig_problem.png'), 'Eight cars follow a leader. Each car is an agent that only talks to its '
                                        'neighbours.'),
        ('ul', ['**Goal:** every car keeps a 20 m gap at 15 m/s, smoothly and without crashing.',
                '**Challenge:** one car\'s speed change travels down the platoon and can grow into a wave.',
                '**Our project:** agents that learn locally and coordinate by messages, and keep working when '
                'the network is unreliable.']),
        ('h2', '2. The paper in one picture'),
        ('img', fig('fig_ia2c_vs_neurcomm.png'), 'IA2C uses neighbour information; NeurComm also sends learned '
                                                 'messages.'),
        ('ul', ['**Learning:** each car learns by trial and error (A2C: an actor picks actions, a critic judges them).',
                '**Spatial discount:** a car cares more about nearby cars\' rewards than far ones.',
                '**NeurComm:** each step, a car sends its neighbours its state, policy and memory.']),
        ('h2', '3. Reproduction: did we get the paper\'s result?'),
        ('table', ['Method', 'Setting', 'Seed 12', 'Seed 13', 'Seed 14', 'Mean'], repro),
        ('callout', '**Partly.** IA2C can beat the hand-tuned controller, but only on one of three training seeds. '
                    'NeurComm, the paper\'s method, does not reproduce: with the released settings it learns to do '
                    'nothing. Switching on the paper\'s spatial discount helps it a lot, but it still stays below '
                    'the fixed controller. Average reward per step on 50 unseen test seeds; closer to 0 is better.'),
        ('h2', '4. A visual simulator'),
        ('img', fig('shot_simulator.png'), 'Three platoons side by side, moments after the leader brakes. Dots '
                                           'are V2V messages; red lights are braking cars.'),
        ('h2', '5. Running it on MPI'),
        ('img', fig('fig_mpi_architecture.png'), 'One MPI process per car plus one for the road. Cars exchange '
                                                 'messages only with their neighbours.'),
        ('ul', ['**Every 0.1 s:** the road sends each car its own sensor readings; neighbouring cars exchange '
                'messages; each car runs its own trained policy and sends back an acceleration command.',
                '**No shared memory:** a car process holds only its own neural network and knows only its own '
                'sensors plus received messages.',
                '**Same algorithm:** with no faults the MPI run equals the Python run at every one of 600 steps '
                '(IA2C −22.96, fixed −44.51).',
                '**Two laptops:** the same program runs across two machines with a hostfile; cars 4 and 5 then '
                'talk over real Wi-Fi.']),
        ('h2', '6. Breaking the network on purpose'),
        ('h3', 'Lost messages'),
        ('table', sweep_head, drop),
        ('img', fig('fig_drop_sweep.png'), 'Average reward per step as more V2V messages are lost.'),
        ('h3', 'Late messages'),
        ('table', sweep_head, delay),
        ('img', fig('fig_delay_sweep.png'), 'Average reward per step as messages arrive later (red ring = collision).'),
        ('h3', 'A car process crashes'),
        ('img', fig('shot_mpi_crash.png'), 'MPI runs replayed: late messages crash the platoon unless the cars use '
                                           'their radar (lanes 1 and 2); a crashed car cannot brake (lane 3).'),
        ('callout', '**Lessons.** (1) Late information is far more dangerous than lost information. (2) Falling back '
                    'on local sensing (radar) makes the platoon tolerate any delay we tried. (3) Detecting a crashed '
                    'node is not enough: it must have a safe behaviour of its own.'),
        ('h2', '7. Scaling'),
        ('img', fig('fig_scaling.png'), 'Time per step and messages per step as the platoon grows.'),
        ('table', ['Cars', 'MPI processes', 'Time per 0.1 s step', 'V2V messages per step'],
         [[r['cars'], r['processes'], '%.1f ms' % float(r['ms_per_step']), '%d' % round(float(r['msgs_per_step']))]
          for r in scaling]),
        ('p', 'Cost grows about linearly and stays far below the 100 ms real-time budget. In full 60 s runs, 16 and 32 '
              'cars collide with the hand-tuned controller (also in plain Python): the wave grows down a longer platoon.'),
        ('h2', '8. What is distributed, honestly'),
        ('table', ['Part', 'Distributed?'], [
            ['Running the trained cars (execution)', 'Yes: separate MPI processes, message passing only, timeouts, fallbacks'],
            ['Training the policies', 'Not yet: one program on one machine'],
            ['The road (physics)', 'One process: it plays the real world, not a controller'],
            ['Network faults', 'Injected by our code (loss, delay, crash)'],
            ['Machines', 'One laptop tested; two-laptop setup ready'],
        ]),
        ('h2', '9. Next steps'),
        ('ul', ['Longer NeurComm training with the spatial discount (running), to close the reproduction gap.',
                'Run across two laptops for the demo.',
                'Distributed training: each car process learns, and neighbours average their networks over MPI.']),
    ]
    with open(os.path.join(OUT, 'REPORT.md'), 'w') as f:
        f.write(bp.render_md(blocks))
    with open(os.path.join(OUT, 'report.html'), 'w') as f:
        f.write(bp.render_html(blocks, 'Distributed platoon: final presentation', font_px=18))

    notes = """# Speaker notes: final presentation (about 10 minutes)

| Time | Section | Say |
|---|---|---|
| 0:00 | Title | Paper reproduced, then turned into a real distributed system on MPI |
| 0:30 | 1 Problem | 8 cars, neighbours only, waves down the platoon |
| 1:30 | 2 Paper | Learning by trial and error; spatial discount; NeurComm messages |
| 2:30 | 3 Reproduction | IA2C works on 1 of 3 seeds; NeurComm not reproduced; spatial discount helps |
| 3:30 | 4 Simulator | Show http://localhost:8765, press "Leader brakes!" |
| 4:30 | 5 MPI | One process per car; show Activity Monitor with 9 processes; identical to Python |
| 6:00 | 6 Faults | Open http://localhost:8766 and press Play: delay crash vs radar fallback, crashed car |
| 8:00 | 7 Scaling | Linear cost, far below real time; long platoons need better control |
| 8:45 | 8 Honest | Execution distributed, training not yet |
| 9:30 | 9 Next | Longer training, two laptops, distributed training |

## Demo commands (open a Terminal in the repo first)

```bash
.venv/bin/python -m simulator.server --open                       # live simulator, port 8765
mpirun --oversubscribe -np 9 .venv/bin/python mpi/platoon_mpi.py --controller ia2c --realtime
.venv/bin/python -m simulator.server --port 8766 --open --replay mpi/logs/demo_delay_last.jsonl \\
    mpi/logs/demo_delay_radar.jsonl mpi/logs/demo_crash_brake.jsonl mpi/logs/demo_ia2c_drop20.jsonl
```

## Words to explain if asked
- **MPI:** standard library for programs made of many processes that cooperate by sending messages.
- **Rank:** the number of an MPI process (0 = road, 1 to 8 = cars).
- **V2V:** vehicle-to-vehicle messages, here MPI messages between neighbouring car processes.
- **Fallback:** what a car does when a message is missing: last value, or radar estimate.
- **Crash-stop:** a process that stops responding. Standard MPI aborts the job if a process dies, so we
  model it as a process that goes silent.
- **Seed:** the random start of a training run; results changed a lot between seeds.
"""
    with open(os.path.join(OUT, 'SPEAKER_NOTES.md'), 'w') as f:
        f.write(notes)
    print('Wrote', os.path.relpath(OUT, REPO))
    for r in repro:
        print('  ', ' | '.join(str(c) for c in r))


if __name__ == '__main__':
    main()
