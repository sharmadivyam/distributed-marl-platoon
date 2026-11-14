"""
Build the short report for a 10-minute presentation (follows the presentation script order:
implementation -> problem -> RL -> A2C -> IA2C/NeurComm -> experiment -> results).

Usage (from repo root):
    python demo/build_presentation_10min.py

Writes demo/presentation_10min/:
    report.html        main deliverable (offline, figures embedded, larger font for screen sharing)
    REPORT.md          same content (VS Code: Cmd+Shift+V)
    SPEAKER_NOTES.md   timings -> what to show, numbers to remember, backup command, likely questions
    fig_*.png          figures used in the report
The full report (all figures and findings) stays in demo/presentation/ (build_presentation.py).
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

DEMO = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(DEMO)
sys.path.insert(0, DEMO)
import build_presentation as bp  # noqa: E402
import make_report as mr  # noqa: E402

OUT = os.path.join(DEMO, 'presentation_10min')
ORANGE = '#eb6834'
BLUE = '#2a78d6'
NEUTRAL = '#d9d8d4'


# ---------------------------------------------------------------- new diagrams
def box(ax, x, y, w, h, title, sub=None, color=BLUE, ink='white'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                facecolor=color, edgecolor='none'))
    if sub:
        ax.text(x, y + 0.13, title, ha='center', va='center', fontsize=10.5, color=ink, fontweight='bold')
        ax.text(x, y - 0.17, sub, ha='center', va='center', fontsize=8.3, color=ink)
    else:
        ax.text(x, y, title, ha='center', va='center', fontsize=10.5, color=ink, fontweight='bold')


def arrow(ax, a, b, color=mr.MUTED, style='-|>', rad=0.0, lw=1.6, ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, color=color, linewidth=lw,
                                 linestyle=ls, connectionstyle='arc3,rad=%s' % rad, shrinkA=0, shrinkB=0))


def fig_rl_loop(path):
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.set_xlim(-0.2, 11.2)
    ax.set_ylim(-1.45, 0.95)
    ax.axis('off')
    w, h = 1.75, 0.8
    items = [('State', 'speed err, gap err,\naccel, …', NEUTRAL, mr.INK),
             ('Policy (actor)', 'neural net\n+ LSTM memory', BLUE, 'white'),
             ('Action', '1 of 4 controller\ngain presets', BLUE, 'white'),
             ('Car simulation', '8 cars move\nfor 0.1 s', NEUTRAL, mr.INK),
             ('Reward', '− penalty\n(closer to 0 = better)', ORANGE, 'white')]
    xs = [0.9 + 2.35 * i for i in range(len(items))]
    for x, (t, sub, c, ink) in zip(xs, items):
        box(ax, x, 0.25, w, h, t, sub, c, ink)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w / 2 + 0.05, 0.25), (b - w / 2 - 0.05, 0.25))
    arrow(ax, (xs[-1], 0.25 - h / 2 - 0.05), (xs[0], 0.25 - h / 2 - 0.05), rad=-0.18)
    ax.text(np.mean([xs[0], xs[-1]]), -1.22, 'new state → repeat every 0.1 s, for every car',
            ha='center', fontsize=9.5, color=mr.MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_comm(path):
    fig, ax = plt.subplots(figsize=(11, 4.1))
    ax.set_xlim(-1.9, 10.4)
    ax.set_ylim(-3.2, 1.0)
    ax.axis('off')
    xs = [1.2, 3.9, 6.6, 9.3]
    names = ['A', 'B', 'C', 'D']
    w, h = 1.15, 0.62
    # IA2C row
    ax.text(-1.8, 0.0, 'IA2C', fontsize=12, fontweight='bold', color=mr.INK, va='center')
    for x, n in zip(xs, names):
        box(ax, x, 0.0, w, h, n)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w / 2 + 0.08, 0.0), (b - w / 2 - 0.08, 0.0), style='<|-|>', color=mr.MUTED)
        ax.text((a + b) / 2, 0.18, 'observes', ha='center', fontsize=8.5, color=mr.MUTED)
    ax.text(xs[0] - w / 2, -0.62, 'Each agent sees its neighbours\' observations; its critic also sees neighbours\' '
            'actions. No messages.', fontsize=9, color=mr.INK)
    # NeurComm row
    y = -2.0
    ax.text(-1.8, y, 'NeurComm', fontsize=12, fontweight='bold', color=mr.INK, va='center')
    for x, n in zip(xs, names):
        box(ax, x, y, w, h, n)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w / 2 + 0.08, y), (b - w / 2 - 0.08, y), style='<|-|>', color=ORANGE, lw=2.2)
        ax.text((a + b) / 2, y + 0.2, 'message', ha='center', fontsize=8.5, color='#b4481f')
    ax.text(xs[0] - w / 2, y - 0.62, 'Every step, each agent sends its neighbours a learned message: '
            'its state + policy + LSTM memory.', fontsize=9, color=mr.INK)
    ax.text(xs[0] - w / 2, y - 0.98, 'A never sends to D directly: information travels hop by hop '
            '(A → B → C → D, one hop per step).', fontsize=9, color='#b4481f')
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def per_car_settings(run_dir, a_map):
    """Most-used gain preset per car in the evaluation, and how often it was used."""
    import glob
    import pandas as pd
    ctl = pd.read_csv(glob.glob(os.path.join(run_dir, 'eva_data', '*_control.csv'))[0])
    acts = np.array([list(map(int, a.split(','))) for a in ctl['action']])
    modal = [int(np.bincount(acts[:, i], minlength=len(a_map)).argmax()) for i in range(acts.shape[1])]
    share = [float((acts[:, i] == modal[i]).mean()) for i in range(acts.shape[1])]
    groups = {}
    for car, a in enumerate(modal):
        groups.setdefault(a, []).append(car + 1)
    text = '; '.join('cars %s use %s' % (', '.join(map(str, cars)), a_map[a]) for a, cars in sorted(groups.items()))
    return text, min(share)


# ---------------------------------------------------------------- main
def main():
    import logging
    logging.disable(logging.INFO)
    os.makedirs(OUT, exist_ok=True)
    fig = lambda name: os.path.join(OUT, name)  # noqa: E731

    R = bp.collect_results()
    env, cfg, best_p = R['env'], R['cfg'], R['best_p']
    base_m, base_s, baseline = R['base_m'], R['base_s'], R['baseline']
    ia2c, nc, presets = R['ia2c'], R['nc'], R['presets']
    long_runs = [m for m in R['methods'][2:]]  # evaluated paper-length NeurComm runs
    mc, ec = cfg['MODEL_CONFIG'], R['env_cfg']
    fixed_name = 'Fixed controller %s' % (best_p[1],)

    def wins(m):
        return int((m['reward'].values > base_m['reward'].values).sum())

    # ---- figures
    bp.fig_task(fig('fig_1_problem.png'))
    fig_rl_loop(fig('fig_2_rl_loop.png'))
    fig_comm(fig('fig_3_ia2c_vs_neurcomm.png'))
    panels = [(fixed_name, R['base_traffic'][R['base_traffic']['episode'] == 1])]
    panels += [(m[0], m[3][m[3]['episode'] == m[3]['episode'].min()]) for m in (ia2c, nc)]
    mr.plot_trajectories(panels, env, 'catchup', fig('fig_4_trajectories.png'))
    if long_runs:
        f, ax = plt.subplots(figsize=(8, 3.8))
        shades = {'PyTorch RMSprop': bp.NC_DARK, 'TF-style RMSprop': mr.COLOR['ma2c_cnet']}
        bp.plot_curves(ax, [('NeurComm, %s' % o[0], shades[o[0]], o[2]) for o in R['overnight']], baseline,
                       'NeurComm training reward over 1,000,000 steps')
        ax.set_ylabel('Avg reward per step\n(higher is better)', color=mr.INK)
        f.tight_layout()
        f.savefig(fig('fig_5_neurcomm_1e6.png'), dpi=170)
        plt.close(f)
    # backup figures
    bp.fig_main_bars([(fixed_name, mr.BASELINE_COLOR, base_s), (ia2c[0], ia2c[1], ia2c[5]), (nc[0], nc[1], nc[5])],
                     fig('fig_b1_bars.png'))
    bp.fig_per_seed(base_m, [(ia2c[0], ia2c[1], ia2c[4]), (nc[0], nc[1], nc[4])], fig('fig_b2_per_seed.png'))
    v1 = [(mr.LABEL[a], mr.COLOR[a], bp.train_curve(os.path.join(DEMO, 'runs_v1_250k', a + '_catchup')))
          for a in ('ia2c', 'ma2c_nc') if os.path.isdir(os.path.join(DEMO, 'runs_v1_250k', a + '_catchup'))]
    if v1:
        f, ax = plt.subplots(figsize=(8, 3.8))
        bp.plot_curves(ax, v1, baseline, 'First 250k-step runs: training reward swings widely')
        ax.set_ylabel('Avg reward per step\n(higher is better)', color=mr.INK)
        f.tight_layout()
        f.savefig(fig('fig_b3_instability.png'), dpi=170)
        plt.close(f)

    # ---- numbers
    ia2c_s, nc_s = ia2c[5], nc[5]
    ia2c_beats = ia2c_s['reward'] > baseline
    ia2c_w, nc_w = wins(ia2c[4]), wins(nc[4])
    steps_main = round(ia2c[7] / 1000)
    pct = 100 * (1 - ia2c_s['reward'] / baseline)
    ia2c_cars, ia2c_min_share = per_car_settings(ia2c[2], env.a_map)
    ia2c_fixed_per_car = ia2c_min_share > 0.95

    res_header = ['Method', 'Avg reward per step (mean ± std)', 'Gap error [m]', 'Speed error [m/s]',
                  'Collisions', 'Better than fixed on']
    res_rows = [[fixed_name + ' (baseline, no learning)', '%.1f ± %.1f' % (baseline, base_s['std']),
                 '%.2f' % base_s['headway_err'], '%.2f' % base_s['speed_err'],
                 '%d / %d' % (base_s['collisions'], base_s['n']), '—']]
    for m, w in ((ia2c, ia2c_w), (nc, nc_w)):
        s = m[5]
        res_rows.append(['**%s**' % m[0] if m is ia2c else m[0], '%.1f ± %.1f' % (s['reward'], s['std']),
                         '%.2f' % s['headway_err'], '%.2f' % s['speed_err'],
                         '%d / %d' % (s['collisions'], s['n']), '%d / %d seeds' % (w, len(m[4]))])

    impl_rows = [
        ['Code and settings', 'The authors\' CACC implementation with the paper\'s configuration files'],
        ['Task', '%s cars, %s s episodes, %s s control step' % (ec['n_vehicle'], ec['episode_length_sec'],
                                                              ec['control_interval_sec'])],
        ['Learner', 'A2C with a %s-unit LSTM and %s-unit fully connected layer' % (mc['num_lstm'], mc['num_fc'])],
        ['Training', '%gk steps per method, one training seed per method' % steps_main],
        ['Model selection', 'checkpoint with the best rolling mean over 5 validation episodes (not simply the last one)'],
    ]
    test_rows = [
        ['Test scenarios', '%d unseen seeds, separate from training and validation' % base_s['n']],
        ['Policy at test time', 'deterministic (always the most likely action)'],
        ['Metrics', 'average reward per step, gap error, speed error, collisions'],
        ['Baseline', 'the 4 fixed controller presets on the same seeds; the best one is the baseline'],
    ]
    preset_rows = [['%s' % (p[1],) + ('  ← baseline' if p is best_p else ''),
                    '%.1f' % p[4]['reward'], '%d / %d' % (p[4]['collisions'], p[4]['n'])] for p in presets]

    long_rows = []
    for m in long_runs:
        s = m[5]
        long_rows.append([m[0].replace('NeurComm 1e6 ', 'NeurComm, 1,000,000 steps '),
                          '%.1f' % s['reward'], '%.2f' % s['headway_err'], '%.2f' % s['speed_err'],
                          '%d / %d' % (s['collisions'], s['n']),
                          '%gk' % round(m[6][1] / 1000) if m[6] else '—'])
    long_do_nothing = bool(long_runs) and all(m[5]['speed_err'] < 0.01 for m in long_runs)

    # ---- report
    blocks = [
        ('h1', 'Multi-agent RL for cooperative cruise control: reproduction'),
        ('p', 'We implemented and reproduced the CACC part of the paper "Multi-agent Reinforcement Learning for '
              'Networked System Control" (Chu, Chinchali & Katti, ICLR 2020). Below: the problem, the RL method, '
              'how we implemented and tested it, and what we found, including why NeurComm did not reproduce '
              'the expected performance.'),
        ('h2', '1. Problem'),
        ('img', fig('fig_1_problem.png'), 'Eight cars follow a leader. Each car is an agent; cars mainly interact '
                                          'with their neighbours.'),
        ('ul', ['**Target:** about **15 m/s** speed and a **20 m** gap to the car ahead.',
                '**Challenge, coordination:** if one car changes speed, the disturbance can propagate down the '
                'platoon.',
                '**Question:** can multi-agent RL control the platoon better than a fixed controller?']),
        ('h2', '2. Reinforcement learning in one picture'),
        ('img', fig('fig_2_rl_loop.png'), 'The RL loop: observe the state, choose an action, receive a reward, '
                                          'repeat.'),
        ('ul', ['**State:** quantities such as speed error, gap information and acceleration.',
                '**Action:** one of **four controller gain settings** for the car-following (OVM) controller, '
                '**not** raw acceleration.',
                '**Reward:** a negative penalty for gap error, speed error and acceleration, with a large '
                'collision penalty. **Closer to zero is better.**']),
        ('h2', '3. The learning method: A2C'),
        ('ul', ['**Actor:** chooses the action.',
                '**Critic:** estimates how good the current situation is.',
                '**Advantage:** after the outcome, was it better or worse than the critic expected? The actor is '
                'pushed towards actions that turned out better than expected.',
                '**LSTM:** gives the agent memory, useful when the current observation does not tell the whole '
                'story.']),
        ('h2', '4. Two multi-agent variants: IA2C vs NeurComm'),
        ('img', fig('fig_3_ia2c_vs_neurcomm.png'), 'IA2C uses neighbour information without messages; NeurComm '
                                                   'adds learned messages between neighbours.'),
        ('h2', '5. How we implemented it'),
        ('table', ['Item', 'Value'], impl_rows),
        ('h2', '6. How we tested it'),
        ('table', ['Item', 'Value'], test_rows),
        ('h3', 'Fixed controller presets on the same %d test seeds' % base_s['n']),
        ('table', ['Preset (α, β)', 'Avg reward per step', 'Collisions'], preset_rows),
        ('h2', '7. Results'),
        ('table', res_header, res_rows),
    ]
    if ia2c_beats:
        blocks.append(('p', '**IA2C beat the fixed baseline:** %.1f vs %.1f average reward per step (%.0f%% less '
                            'penalty), better on %d of %d seeds, with %d collisions.'
                       % (ia2c_s['reward'], baseline, pct, ia2c_w, len(ia2c[4]), ia2c_s['collisions'])))
    if nc_s['speed_err'] < base_s['speed_err'] and nc_s['headway_err'] > base_s['headway_err']:
        blocks.append(('callout', '**Key interpretation:** NeurComm\'s low speed error (%.2f m/s) does **not** mean '
                                  'it performed well. It kept speed close to 15 m/s but failed to correct the 20 m '
                                  'spacing (gap error %.2f m), which hurt the total reward.'
                       % (nc_s['speed_err'], nc_s['headway_err'])))
    blocks += [
        ('img', fig('fig_4_trajectories.png'), 'One unseen test episode. Top: gap of each car; bottom: speed. '
                                               'Dotted line = target.'),
        ('ul', ['**Fixed controller:** the disturbance grows as it travels down the platoon, and later cars '
                'take longer to settle.',
                '**IA2C:** the disturbance shrinks down the platoon and settles sooner. %s'
                % ('What it learned is a per-car combination of settings (%s) that works better than giving '
                   'every car the same setting.' % ia2c_cars if ia2c_fixed_per_car else
                   'Its settings per car: %s.' % ia2c_cars),
                '**NeurComm:** car 1 closes its gap, but car 2 does not respond properly, so the large gap simply '
                'moves back to car 2.']),
    ]
    if long_runs:
        blocks += [('h2', '8. NeurComm with the paper\'s full training length'),
                   ('table', ['Run', 'Avg reward per step', 'Gap error [m]', 'Speed error [m/s]', 'Collisions',
                              'Best checkpoint at'], long_rows)]
        if long_do_nothing:
            blocks.append(('p', 'With 1,000,000 training steps (with either optimizer version) the behaviour became '
                                'even more complete: the cars essentially stayed at 15 m/s and car 1 never closed '
                                'its gap (speed error 0.00). Longer training did not fix it.'))
        blocks.append(('img', fig('fig_5_neurcomm_1e6.png'), 'NeurComm training reward stays below the fixed '
                                                             'baseline (dashed) for the whole run.'))
    n = 9 if long_runs else 8
    blocks += [
        ('h2', '%d. Why NeurComm performed poorly, and what it means' % n),
        ('ul', ['**Observed failure mode, a "do nothing" local optimum:** not accelerating never causes a crash, '
                'but it never corrects the gap. That gives low speed error but high gap error, so the total '
                'reward stays poor.',
                '**Training was unstable.** We tested a TensorFlow-style RMSprop variant (the paper\'s original '
                'optimizer). It reduced some crash-like training behaviour, but the final NeurComm policy remained '
                'poor.',
                '**No single proven root cause:** with one training seed we can report what we observed, not '
                'prove that communication itself is the cause.',
                '**What it means:** adding communication does not automatically produce better coordination. It '
                'gives a richer model, but the learning problem can also become harder. IA2C learned useful '
                'coordination in our reproduction; NeurComm did not reproduce the paper\'s benefit.']),
        ('h2', '%d. Limitations' % (n + 1)),
        ('ul', ['**One training seed per method:** more seeds are needed before making a strong statistical claim.',
                '**NeurComm sensitivity:** spatial discounting, entropy and learning rate still need to be studied.',
                '**One process:** all agents are simulated in a single process; truly distributed execution, with '
                'one process per car, is future work.']),
        ('h2', '%d. Summary' % (n + 2)),
        ('callout', 'Implement the paper → train the agents → select the best validation checkpoint → test on %d '
                    'unseen seeds → compare with the best fixed controller.' % base_s['n']),
        ('p', '%s NeurComm did not; it converged towards a do-nothing solution. That is our main reproduction '
              'finding and the main question for further investigation.'
         % ('IA2C improved substantially over the baseline.' if ia2c_beats else
            'IA2C did not beat the baseline.')),
        ('hr',),
        ('h2', 'Backup (only if asked)'),
        ('img', fig('fig_b1_bars.png'), 'The main results as bars (95% confidence intervals of the mean).'),
        ('img', fig('fig_b2_per_seed.png'), 'Per-seed comparison: each dot is one test episode; below the line = the '
                                            'learned method was better.'),
    ]
    if v1:
        blocks.append(('img', fig('fig_b3_instability.png'), 'Why we select the best validation checkpoint: in our '
                                                             'first 250k-step runs, training reward swung widely.'))

    with open(os.path.join(OUT, 'REPORT.md'), 'w') as f:
        f.write(bp.render_md(blocks))
    with open(os.path.join(OUT, 'report.html'), 'w') as f:
        f.write(bp.render_html(blocks, 'CACC reproduction: 10-minute presentation', font_px=18))

    # ---- speaker notes
    ep1 = ia2c[4]['reward'].values[0]
    lines = [
        '# Speaker notes: 10-minute presentation', '',
        '## Before the meeting', '',
        '- Open `demo/presentation_10min/report.html` in the browser (double-click it in Finder).',
        '- Open a terminal in the repo with the backup command (below) typed in, but don\'t run it.',
        '- Don\'t train live. 10 minutes is too short, and the report already has the results.', '',
        '## Timeline: what to show', '',
        '| Time | Script block | Show in report.html |', '|---|---|---|',
        '| 0:00–0:25 | What we did | Title + first paragraph |',
        '| 0:25–1:20 | Problem | Section 1 (8-car figure) |',
        '| 1:20–2:15 | RL from zero | Section 2 (RL loop figure) |',
        '| 2:15–3:00 | A2C | Section 3 (no equations) |',
        '| 3:00–3:45 | IA2C vs NeurComm | Section 4 (A ↔ B ↔ C ↔ D figure) |',
        '| 3:45–4:35 | How we implemented it | Section 5 table |',
        '| 4:35–5:20 | How we tested it | Section 6 + presets table |',
        '| 5:20–6:20 | Results | Section 7: table, key interpretation, trajectory figure |',
        '| 6:20–7:35 | Why NeurComm performed poorly | Section 8 (1e6 table + curve), then section 9 |',
        '| 7:35–8:25 | What it means | Section 9, last bullet |',
        '| 8:25–9:15 | Limitations | Section 10 (only these 3) |',
        '| 9:15–10:00 | Closing | Section 11 |', '',
        '## Numbers to remember', '',
        '- Baseline **%.1f** · IA2C **%.1f** · NeurComm **%.1f** · **%d** test seeds · **%gk** training steps'
        % (baseline, ia2c_s['reward'], nc_s['reward'], base_s['n'], steps_main),
        '- IA2C: %.2f m gap error, %.2f m/s speed error, %d/%d collisions, better on %d/%d seeds'
        % (ia2c_s['headway_err'], ia2c_s['speed_err'], ia2c_s['collisions'], ia2c_s['n'], ia2c_w, len(ia2c[4])),
        '- NeurComm: %.2f m gap error, %.2f m/s speed error, %d/%d collisions'
        % (nc_s['headway_err'], nc_s['speed_err'], nc_s['collisions'], nc_s['n']),
    ]
    if long_runs:
        lines.append('- NeurComm at 1,000,000 steps: %s' % ', '.join('%.1f' % m[5]['reward'] for m in long_runs))
    lines += [
        '', '## Wording to watch', '',
        '- Say **"average reward per step"**, not "cumulative reward". That\'s what the table reports.',
        '- **Fixed controller** = the non-learning car-following (OVM) controller with one fixed gain preset. '
        'It is the baseline.',
        '- **Action** = choosing 1 of 4 gain presets, not acceleration.',
        '- **Hidden state** = the LSTM\'s internal memory vector, not the physical car state.',
        '- **Reward** is a negative penalty, so closer to zero is better.', '',
        '## If you get stuck (one sentence)', '',
        '> We reproduced the CACC setup, trained IA2C and NeurComm, selected the best validation checkpoint, '
        'tested on %d unseen seeds against the best fixed controller, and found that IA2C learned useful '
        'coordination while NeurComm converged to a poor do-nothing local optimum.' % base_s['n'], '',
        '## Backup: "does the code actually run?" (~20 s)', '',
        'Evaluates the saved IA2C model on one unseen seed. It works on a **copy**, so the saved results are '
        'never touched:', '',
        '```bash',
        'rm -rf /tmp/ia2c_live && cp -R demo/runs/ia2c_catchup /tmp/ia2c_live',
        '.venv/bin/python main.py --base-dir /tmp/ia2c_live evaluate --evaluation-seeds 100000 --checkpoint best',
        '```', '',
        'Expected last line: `test 0, avg reward %.2f` (the same seed as the trajectory figure).' % ep1, '',
        '## Likely questions', '',
        '| Question | Short answer |', '|---|---|',
        '| Why is the best IA2C checkpoint at %gk, not %gk steps? | RL training is noisy; we keep the checkpoint with '
        'the best validation reward. Validation seeds never overlap the test seeds. |'
        % (round(ia2c[6][1] / 1000, 1) if ia2c[6] else 0, steps_main),
        '| Isn\'t picking the best checkpoint cheating? | No. It is chosen on separate validation episodes; every '
        'number in the table comes from %d different, unseen seeds. |' % base_s['n'],
        '| What did IA2C actually learn? | %s. %s The trajectory figure shows this mix damps the disturbance '
        'down the platoon, which no single uniform setting does. |'
        % (ia2c_cars[0].upper() + ia2c_cars[1:],
           'At test time each car essentially keeps its setting (deterministic policy), so it learned *which '
           'car should use which setting*, not switching over time.' if ia2c_fixed_per_car else ''),
        '| Could you get the same with a fixed per-car combination? | Possibly. Our baseline only tries the same '
        'setting for every car (4 options). Searching all 4^8 per-car combinations would be a stronger baseline; '
        'that is a fair next step. |',
        '| Why not just train NeurComm longer? | We did: 1,000,000 steps (the paper\'s length) still converged to '
        'do nothing. |',
        '| Is it really distributed? | Each agent uses only neighbour information, but all agents run in one '
        'process. One process per car is future work. |',
        '| How sure are you? | One training seed per method, so this is what we observed, not a statistical proof. '
        'More seeds are the next step. |',
    ]
    with open(os.path.join(OUT, 'SPEAKER_NOTES.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('Wrote %s' % os.path.relpath(OUT, REPO))


if __name__ == '__main__':
    main()
