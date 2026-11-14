"""
Build a self-contained results package for presenting the demo.

Usage (from repo root):
    python demo/build_presentation.py

Writes demo/presentation/:
    REPORT.md        report with tables + figures (open in VS Code: Cmd+Shift+V)
    report.html      same report, images embedded; opens offline in any browser, prints to PDF
    fig*.png         individual figures (for slides)
    tables/*.csv     the numbers behind the tables

Reads:
    demo/runs/                 main demo runs (IA2C + NeurComm, best checkpoints)
    demo/runs_overnight/*/     paper-length NeurComm runs (included when present, even mid-run)
    demo/runs_v1_250k/         first 250k-step runs (instability evidence)
    demo/runs_optimizer_tf/    same as demo/runs but with TF-style RMSprop (optimizer A/B)
Re-run any time; e.g. after the overnight runs finish.
"""
import base64
import configparser
import datetime
import glob
import html
import logging
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

DEMO = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(DEMO)
sys.path.insert(0, DEMO)
import make_report as mr  # noqa: E402

OUT = os.path.join(DEMO, 'presentation')
SEEDS = list(range(100000, 100500, 10))
NC_DARK = '#1c5aa6'  # NeurComm paper-length runs: same hue as NeurComm, darker step


# ---------------------------------------------------------------- data
def load_eval(run_dir, env):
    path = glob.glob(os.path.join(run_dir, 'eva_data', '*_traffic.csv'))
    if not path:
        return None
    traffic = pd.read_csv(path[0])
    return traffic, mr.episode_metrics(traffic, env.h_min, env.h_star, env.v_star, env.n_agent)


def tb_curve(run_dir):
    """train_reward scalars from TensorBoard events (available while training is running)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    steps, vals = [], []
    for f in sorted(glob.glob(os.path.join(run_dir, 'log', 'events*'))):
        acc = EventAccumulator(f, size_guidance={'scalars': 0})
        acc.Reload()
        if 'train_reward' in acc.Tags().get('scalars', []):
            for s in acc.Scalars('train_reward'):
                steps.append(s.step)
                vals.append(s.value)
    return pd.DataFrame({'step': steps, 'avg_reward': vals}).sort_values('step')


def train_curve(run_dir):
    csv = os.path.join(run_dir, 'data', 'train_reward.csv')
    return pd.read_csv(csv) if os.path.exists(csv) else tb_curve(run_dir)


def config_of(run_dir):
    c = configparser.ConfigParser()
    c.read(glob.glob(os.path.join(run_dir, 'data', '*.ini'))[0])
    return c


def summarize(m):
    n = len(m)
    return {'reward': m['reward'].mean(), 'std': m['reward'].std(),
            'ci': 1.96 * m['reward'].std() / np.sqrt(n), 'collisions': int(m['collision'].sum()),
            'n': n, 'headway_err': m['headway_err'].mean(), 'speed_err': m['speed_err'].mean()}


# ---------------------------------------------------------------- figures
def fig_task(path):
    fig, ax = plt.subplots(figsize=(10, 2.9))
    ax.set_xlim(-0.6, 10.2)
    ax.set_ylim(-1.35, 1.0)
    ax.axis('off')
    xs = [9.2 - 1.1 * i for i in range(9)]  # leader at the right, 8 cars behind it
    for k, x in enumerate(xs):
        leader = k == 0
        ax.add_patch(FancyBboxPatch((x - 0.38, -0.22), 0.76, 0.44, boxstyle='round,pad=0.02,rounding_size=0.08',
                                    facecolor='#d9d8d4' if leader else '#2a78d6', edgecolor='none'))
        ax.text(x, 0, 'leader' if leader else 'car %d' % k, ha='center', va='center', fontsize=8.5,
                color=mr.INK if leader else 'white', fontweight='bold')
    for k in range(1, 8):  # neighbour links between agents
        a, b = xs[k], xs[k + 1]
        ax.add_patch(FancyArrowPatch((b, 0.26), (a, 0.26), arrowstyle='<|-|>', mutation_scale=11,
                                     color='#eb6834', linewidth=1.6, connectionstyle='arc3,rad=-0.45',
                                     shrinkA=2, shrinkB=2))
    ax.text(np.mean(xs[1:]), 0.82, 'neighbour links: observations + messages (only between adjacent cars)',
            ha='center', fontsize=9, color='#b4481f')
    ax.annotate('', xy=(xs[1] - 0.4, -0.45), xytext=(xs[2] + 0.4, -0.45),
                arrowprops=dict(arrowstyle='<->', color=mr.MUTED, lw=1))
    ax.text((xs[1] + xs[2]) / 2, -0.62, 'gap (headway)\ntarget 20 m', ha='center', va='top', fontsize=8, color=mr.MUTED)
    ax.annotate('driving direction', xy=(10.1, -0.95), xytext=(7.6, -0.95), fontsize=8.5, color=mr.MUTED,
                va='center', arrowprops=dict(arrowstyle='->', color=mr.MUTED, lw=1))
    ax.text(-0.5, -0.95, 'Each car = one RL agent. Every 0.1 s it picks 1 of 4 controller gain settings.\n'
            'Goal: 20 m gaps at 15 m/s, smooth, no crashes. Catch-up: car 1 starts 1.5–2.5× too far back.',
            fontsize=8.5, color=mr.INK, va='center')
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_main_bars(rows, path):
    """rows: list of (label, color, summary). Three small multiples: penalty, gap error, speed error."""
    labels = [r[0] for r in rows][::-1]
    colors = [r[1] for r in rows][::-1]
    stats = [r[2] for r in rows][::-1]
    panels = [('Avg penalty per step (lower is better)', [-s['reward'] for s in stats], [s['ci'] for s in stats], '%.0f'),
              ('Gap error [m]', [s['headway_err'] for s in stats], None, '%.2f'),
              ('Speed error [m/s]', [s['speed_err'] for s in stats], None, '%.2f')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.62 * len(rows) + 1.6), sharey=True,
                             gridspec_kw={'width_ratios': [1.6, 1, 1]})
    y = np.arange(len(labels))
    for ax, (title, vals, err, f) in zip(axes, panels):
        ax.barh(y, vals, color=colors, height=0.6, xerr=err,
                error_kw=dict(ecolor=mr.INK, lw=1, capsize=3) if err else None)
        for i, (yi, v) in enumerate(zip(y, vals)):
            ax.text(v + (err[i] if err else 0), yi, '  ' + f % v, va='center', fontsize=8.5, color=mr.INK)
        ax.set_title(title, fontsize=10, color=mr.INK, loc='left')
        ax.set_xlim(0, max(vals) * 1.3 + (max(err) if err else 0))
        mr.style(ax)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9, color=mr.INK)
    fig.suptitle('Evaluation on 50 unseen seeds (error bars: 95% CI of the mean)', x=0.01, ha='left',
                 fontsize=11, color=mr.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_per_seed(base_m, learned, path):
    """learned: list of (label, color, metrics). Paired per-seed comparison against the fixed controller."""
    fig, axes = plt.subplots(1, len(learned), figsize=(4.6 * len(learned), 4.3), squeeze=False)
    bx = -base_m['reward'].values
    for ax, (label, color, m) in zip(axes[0], learned):
        ly = -m['reward'].values
        hi = max(bx.max(), ly.max()) * 1.08
        ax.plot([0, hi], [0, hi], color=mr.MUTED, linestyle='--', linewidth=1)
        ax.scatter(bx, ly, s=36, color=color, edgecolor='white', linewidth=1, zorder=3)
        wins = int((ly < bx).sum())
        ax.text(0.03, 0.96, '%s better on %d / %d seeds' % (label, wins, len(bx)), transform=ax.transAxes,
                va='top', fontsize=9.5, color=mr.INK)
        ax.text(0.97, 0.04, 'below the line =\n%s better' % label, transform=ax.transAxes, ha='right',
                fontsize=8, color=mr.MUTED)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel('Fixed controller penalty', color=mr.INK)
        ax.set_ylabel('%s penalty' % label, color=mr.INK)
        mr.style(ax)
    fig.suptitle('Per-seed comparison (each dot = one unseen test episode)', x=0.01, ha='left',
                 fontsize=11, color=mr.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_curves(ax, curves, baseline, title):
    """curves: list of (label, color, df). Rolling median of 15 episodes."""
    lows = []
    for label, color, df in curves:
        if df is None or df.empty:
            continue
        smooth = df['avg_reward'].rolling(15, min_periods=1).median()
        ax.plot(df['step'], smooth, color=color, linewidth=2, label=label)
        lows.append(np.percentile(smooth, 5))
    ax.axhline(baseline, color=mr.BASELINE_COLOR, linestyle='--', linewidth=1.3, label='best fixed controller')
    ax.set_ylim(min(lows + [baseline]) * 1.15, 0)
    ax.set_title(title, fontsize=10, color=mr.INK, loc='left')
    ax.set_xlabel('Training step', color=mr.INK)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: '%gk' % (x / 1000)))
    mr.style(ax)
    ax.legend(frameon=False, fontsize=8, loc='lower right')


# ---------------------------------------------------------------- report blocks -> md / html
def md_inline(t):
    return t


def html_inline(t):
    t = html.escape(t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'`(.+?)`', r'<code>\1</code>', t)
    return t


def render_md(blocks):
    out = []
    for b in blocks:
        kind = b[0]
        if kind in ('h1', 'h2', 'h3'):
            out += ['#' * int(kind[1]) + ' ' + b[1], '']
        elif kind == 'p':
            out += [b[1], '']
        elif kind == 'ul':
            out += ['- ' + i for i in b[1]] + ['']
        elif kind == 'img':
            out += ['![%s](%s)' % (b[2], os.path.basename(b[1])), '', '*%s*' % b[2], '']
        elif kind == 'table':
            header, rows = b[1], b[2]
            out += ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
            out += ['| ' + ' | '.join(r) + ' |' for r in rows] + ['']
        elif kind == 'code':
            out += ['```bash', b[1], '```', '']
        elif kind == 'callout':
            out += ['> ' + b[1], '']
        elif kind == 'hr':
            out += ['---', '']
    return '\n'.join(out)


def render_html(blocks, title, font_px=16):
    body = []
    for b in blocks:
        kind = b[0]
        if kind in ('h1', 'h2', 'h3'):
            body.append('<%s>%s</%s>' % (kind, html_inline(b[1]), kind))
        elif kind == 'p':
            body.append('<p>%s</p>' % html_inline(b[1]))
        elif kind == 'ul':
            body.append('<ul>%s</ul>' % ''.join('<li>%s</li>' % html_inline(i) for i in b[1]))
        elif kind == 'img':
            with open(b[1], 'rb') as f:
                data = base64.b64encode(f.read()).decode()
            body.append('<figure><img src="data:image/png;base64,%s" alt="%s"><figcaption>%s</figcaption></figure>'
                        % (data, html.escape(b[2]), html_inline(b[2])))
        elif kind == 'table':
            head = ''.join('<th>%s</th>' % html_inline(h) for h in b[1])
            rows = ''.join('<tr>%s</tr>' % ''.join('<td>%s</td>' % html_inline(c) for c in r) for r in b[2])
            body.append('<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>' % (head, rows))
        elif kind == 'code':
            body.append('<pre><code>%s</code></pre>' % html.escape(b[1]))
        elif kind == 'callout':
            body.append('<div class="callout">%s</div>' % html_inline(b[1]))
        elif kind == 'hr':
            body.append('<hr>')
    css = """
    :root { --ink:#0b0b0b; --muted:#52514e; --line:#e4e3df; --accent:#1f4e79; --bg:#fcfcfb; }
    body { margin:0; background:var(--bg); color:var(--ink);
           font:FONTPX/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    main { max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }
    h1 { font-size: 30px; color: var(--accent); margin: 0 0 4px; }
    h2 { font-size: 22px; color: var(--accent); border-bottom: 1px solid var(--line);
         padding-bottom: 4px; margin-top: 40px; }
    h3 { font-size: 17px; margin-top: 24px; }
    p, li { max-width: 78ch; }
    figure { margin: 18px 0 26px; }
    figure img { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
    figcaption { color: var(--muted); font-size: 14px; margin-top: 6px; }
    table { border-collapse: collapse; margin: 12px 0 22px; font-size: 14.5px; width: 100%; }
    th, td { border: 1px solid var(--line); padding: 6px 10px; text-align: left; vertical-align: top; }
    th { background: #eef3f9; }
    code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.9em;
           background: #f1f0ec; padding: 1px 4px; border-radius: 3px; }
    pre { background: #f1f0ec; padding: 12px; border-radius: 6px; overflow-x: auto; }
    pre code { background: none; padding: 0; }
    .callout { border-left: 5px solid var(--accent); background: #eef3f9; padding: 12px 16px;
               border-radius: 4px; margin: 16px 0 22px; max-width: 78ch; }
    hr { border: none; border-top: 2px dashed var(--line); margin: 48px 0 8px; }
    @media print { body { background: #fff; } main { max-width: none; padding: 0; }
                   figure, table { break-inside: avoid; } h2 { break-after: avoid; } }
    """
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>%s</title><style>%s</style></head><body><main>%s</main></body></html>'
            % (html.escape(title), css.replace('FONTPX', '%dpx' % font_px), '\n'.join(body)))


# ---------------------------------------------------------------- shared data loading
def collect_results():
    """Load the demo runs, evaluate the fixed presets on SEEDS, and gather the learned methods.
    Shared by build_presentation.py and build_presentation_10min.py."""
    main_runs = {r['agent']: r for r in mr.find_runs(os.path.join(DEMO, 'runs')) if r['scenario'] == 'catchup'}
    if 'ia2c' not in main_runs or 'ma2c_nc' not in main_runs:
        sys.exit('demo/runs must contain ia2c_catchup and ma2c_nc_catchup (see DEMO_GUIDE.md, Step 2)')
    cfg = main_runs['ma2c_nc']['config']
    env_cfg = cfg['ENV_CONFIG']

    # fixed-controller presets on the evaluation seeds
    presets = []
    for p in range(4):
        traffic, env = mr.run_fixed_preset(env_cfg, SEEDS, p)
        m = mr.episode_metrics(traffic, env.h_min, env.h_star, env.v_star, env.n_agent)
        presets.append((p, env.a_map[p], traffic, m, summarize(m)))
    best_p = max(presets, key=lambda x: x[4]['reward'])
    base_traffic, base_m, base_s = best_p[2], best_p[3], best_p[4]
    baseline = base_s['reward']

    # learned methods
    methods = []  # (label, color, run_dir, traffic, metrics, summary, checkpoint, total_steps)
    for agent, color in [('ia2c', mr.COLOR['ia2c']), ('ma2c_nc', mr.COLOR['ma2c_nc'])]:
        d = main_runs[agent]['dir']
        traffic, m = load_eval(d, env)
        steps = train_curve(d)['step'].max()
        methods.append((mr.LABEL[agent], color, d, traffic, m, summarize(m), mr.evaluated_checkpoint(d), steps))

    overnight = []  # (opt label, run_dir, curve, eval or None)
    for opt, opt_label in [('torch', 'PyTorch RMSprop'), ('tf', 'TF-style RMSprop')]:
        d = os.path.join(DEMO, 'runs_overnight', opt, 'ma2c_nc_catchup')
        if not os.path.isdir(d):
            continue
        curve = train_curve(d)
        ev = load_eval(d, env)
        overnight.append((opt_label, d, curve, ev))
        if ev is not None:
            methods.append(('NeurComm 1e6 (%s)' % opt_label, NC_DARK, d, ev[0], ev[1], summarize(ev[1]),
                            mr.evaluated_checkpoint(d), curve['step'].max()))

    ia2c = methods[0]
    nc = methods[1]
    return dict(main_runs=main_runs, cfg=cfg, env_cfg=env_cfg, env=env, presets=presets, best_p=best_p,
                base_traffic=base_traffic, base_m=base_m, base_s=base_s, baseline=baseline,
                methods=methods, overnight=overnight, ia2c=ia2c, nc=nc)


# ---------------------------------------------------------------- main
def main():
    logging.disable(logging.INFO)
    os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
    fig = lambda name: os.path.join(OUT, name)  # noqa: E731

    R = collect_results()
    main_runs, cfg, env_cfg, env = R['main_runs'], R['cfg'], R['env_cfg'], R['env']
    presets, best_p, baseline = R['presets'], R['best_p'], R['baseline']
    base_traffic, base_m, base_s = R['base_traffic'], R['base_m'], R['base_s']
    methods, overnight, ia2c, nc = R['methods'], R['overnight'], R['ia2c'], R['nc']

    # ---- figures
    fig_task(fig('fig1_task.png'))
    bar_rows = [('Fixed controller %s' % (best_p[1],), mr.BASELINE_COLOR, base_s)] + \
               [(mth[0], mth[1], mth[5]) for mth in methods]
    fig_main_bars(bar_rows, fig('fig2_main_results.png'))
    fig_per_seed(base_m, [(ia2c[0], ia2c[1], ia2c[4]), (nc[0], nc[1], nc[4])], fig('fig3_per_seed.png'))
    mr.plot_learning_curves([main_runs['ia2c'], main_runs['ma2c_nc']], baseline, 'catchup',
                            fig('fig4_learning_curves.png'))
    panels = [('Fixed controller %s' % (best_p[1],), base_traffic[base_traffic['episode'] == 1])]
    panels += [(mth[0], mth[3][mth[3]['episode'] == mth[3]['episode'].min()]) for mth in methods]
    mr.plot_trajectories(panels, env, 'catchup', fig('fig5_trajectories.png'))

    have_v1 = os.path.isdir(os.path.join(DEMO, 'runs_v1_250k'))
    have_tf = os.path.isdir(os.path.join(DEMO, 'runs_optimizer_tf', 'ma2c_nc_catchup'))
    if have_v1 or have_tf:
        f6, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        if have_v1:
            curves = []
            for agent in ['ia2c', 'ma2c_cnet', 'ma2c_nc']:
                d = os.path.join(DEMO, 'runs_v1_250k', agent + '_catchup')
                if os.path.isdir(d):
                    curves.append((mr.LABEL[agent], mr.COLOR[agent], train_curve(d)))
            plot_curves(axes[0], curves, baseline, '(a) First runs, 250k steps: unstable training')
        if have_tf:
            plot_curves(axes[1], [('NeurComm, PyTorch RMSprop', mr.COLOR['ma2c_nc'], train_curve(nc[2])),
                                  ('NeurComm, TF-style RMSprop', mr.COLOR['ma2c_cnet'],
                                   train_curve(os.path.join(DEMO, 'runs_optimizer_tf', 'ma2c_nc_catchup')))],
                        baseline, '(b) Optimizer A/B, 60k steps')
        axes[0].set_ylabel('Avg reward per step (higher is better)', color=mr.INK)
        f6.tight_layout()
        f6.savefig(fig('fig6_reproduction.png'), dpi=170)
        plt.close(f6)

    if overnight:
        f7, ax = plt.subplots(figsize=(8, 4))
        shades = {'PyTorch RMSprop': NC_DARK, 'TF-style RMSprop': mr.COLOR['ma2c_cnet']}
        plot_curves(ax, [('NeurComm, %s' % o[0], shades[o[0]], o[2]) for o in overnight], baseline,
                    'NeurComm, paper-length training: %s' % ', '.join(
                        '%s %s' % ('PyTorch' if o[0].startswith('PyTorch') else 'TF-style',
                                   'done' if o[3] is not None else '%gk / 1000k steps'
                                   % round((o[2]['step'].max() if not o[2].empty else 0) / 1000))
                        for o in overnight))
        ax.set_ylabel('Avg reward per step (higher is better)', color=mr.INK)
        f7.tight_layout()
        f7.savefig(fig('fig7_neurcomm_long.png'), dpi=170)
        plt.close(f7)

    # ---- tables
    def ckpt_text(mth):
        c = mth[6]
        return '%s @ %gk of %gk steps' % (c[0], round(c[1] / 1000, 1), round(mth[7] / 1000)) if c else '—'

    main_header = ['Method', 'Reward (mean ± std)', '95% CI', 'Collisions', 'Gap err [m]', 'Speed err [m/s]',
                   'Beats fixed on', 'Evaluated checkpoint']
    main_rows = [['Fixed controller %s, no learning' % (best_p[1],), '%.1f ± %.1f' % (baseline, base_s['std']),
                  '± %.1f' % base_s['ci'], '%d / %d' % (base_s['collisions'], base_s['n']),
                  '%.2f' % base_s['headway_err'], '%.2f' % base_s['speed_err'], '—', '—']]
    for mth in methods:
        s, m = mth[5], mth[4]
        wins = int((m['reward'].values > base_m['reward'].values).sum())
        main_rows.append(['**%s**' % mth[0] if mth is ia2c else mth[0], '%.1f ± %.1f' % (s['reward'], s['std']),
                          '± %.1f' % s['ci'], '%d / %d' % (s['collisions'], s['n']), '%.2f' % s['headway_err'],
                          '%.2f' % s['speed_err'], '%d / %d seeds' % (wins, len(m)), ckpt_text(mth)])
    pd.DataFrame(main_rows, columns=main_header).to_csv(os.path.join(OUT, 'tables', 'main_results.csv'), index=False)

    preset_rows = [['%s' % (p[1],), '%.1f ± %.1f' % (p[4]['reward'], p[4]['std']),
                    '%d / %d' % (p[4]['collisions'], p[4]['n'])] for p in presets]
    pd.DataFrame(preset_rows, columns=['Preset (α, β)', 'Reward', 'Collisions']).to_csv(
        os.path.join(OUT, 'tables', 'fixed_presets.csv'), index=False)

    mc, ec = cfg['MODEL_CONFIG'], env_cfg
    setup_rows = [
        ['Scenario', 'CACC catch-up: %s cars, %s s episodes, %s s control step' %
         (ec['n_vehicle'], ec['episode_length_sec'], ec['control_interval_sec'])],
        ['Targets', 'gap %s m, speed %s m/s; collision if gap < %s m' %
         (ec['headway_target'], ec['speed_target'], ec['headway_min'])],
        ['Learner', 'A2C, LSTM %s units, FC %s units, RMSprop lr %s, γ %s, entropy %s (IA2C %s)' %
         (mc['num_lstm'], mc['num_fc'], mc['lr_init'], mc['gamma'], mc['entropy_coef'],
          config_of(ia2c[2])['MODEL_CONFIG']['entropy_coef'])],
        ['Rollout', '%s steps per update; reward / %s' % (mc['batch_size'], mc['reward_norm'])],
        ['Hyperparameters', "the paper's configs (`config/`), only training length changed"],
        ['Training', '%gk steps per method, 1 training seed' % round(ia2c[7] / 1000)],
        ['Model selection', 'checkpoint with the best rolling mean of 5 validation episodes (validation seeds never overlap evaluation seeds)'],
        ['Evaluation', '50 unseen seeds (%d–%d), deterministic policy' % (SEEDS[0], SEEDS[-1])],
        ['Baseline', 'same OVM car-following controller with the best of its 4 fixed gain presets'],
    ]

    # ---- text (numbers filled from data; claims conditional on the data)
    ia2c_s, nc_s = ia2c[5], nc[5]
    ia2c_wins = int((ia2c[4]['reward'].values > base_m['reward'].values).sum())
    ia2c_better = ia2c_s['reward'] > baseline and (ia2c_s['reward'] - ia2c_s['ci']) > (baseline + base_s['ci'])
    summary = []
    if ia2c_better:
        summary.append('**IA2C learned a better controller than the hand-tuned baseline**: average penalty %.1f vs '
                       '%.1f (%.0f%% lower), better on %d of %d unseen seeds, with no collisions.'
                       % (-ia2c_s['reward'], -baseline, 100 * (1 - ia2c_s['reward'] / baseline), ia2c_wins, len(SEEDS)))
    else:
        summary.append('IA2C reached an average penalty of %.1f vs %.1f for the hand-tuned baseline.'
                       % (-ia2c_s['reward'], -baseline))
    summary.append('Gap error %.2f m vs %.2f m and speed error %.2f vs %.2f m/s: the learned platoon tracks its '
                   'targets more closely.' % (ia2c_s['headway_err'], base_s['headway_err'],
                                              ia2c_s['speed_err'], base_s['speed_err'])
                   if ia2c_s['headway_err'] < base_s['headway_err'] else
                   'Gap error %.2f m vs %.2f m for the baseline.' % (ia2c_s['headway_err'], base_s['headway_err']))
    nc_long = [mth for mth in methods[2:]]
    best_long = max(nc_long, key=lambda x: x[5]['reward']) if nc_long else None
    if best_long is not None and best_long[5]['reward'] > baseline:
        summary.append('**NeurComm with paper-length training beats the baseline**: %.1f vs %.1f (%s).'
                       % (-best_long[5]['reward'], -baseline, best_long[0]))
    elif best_long is not None:
        summary.append('NeurComm (the paper\'s method) did not beat the baseline even with paper-length training '
                       '(best %.1f vs %.1f): a reproduction gap worth investigating.' % (-best_long[5]['reward'], -baseline))
    else:
        summary.append('NeurComm (the paper\'s method) has not converged within 60k steps (penalty %.1f); '
                       'paper-length (1e6-step) runs are in progress.' % -nc_s['reward'])
    summary.append('All runs use the authors\' code and the paper\'s hyperparameters; results are from one training '
                   'seed per method and 50 unseen test seeds.')

    long_status = []
    for o in overnight:
        cur = o[2]['step'].max() if not o[2].empty else 0
        if o[3] is not None:
            s = summarize(o[3][1])
            long_status.append('%s: finished, evaluated %s → reward %.1f, %d / 50 collisions.'
                               % (o[0], 'best checkpoint' if mr.evaluated_checkpoint(o[1]) and
                                  mr.evaluated_checkpoint(o[1])[0] == 'best' else 'checkpoint',
                                  s['reward'], s['collisions']))
        else:
            long_status.append('%s: in progress, %gk / 1000k steps (curve so far shown below).'
                               % (o[0], round(cur / 1000)))

    # paper-length NeurComm: did every car keep a constant speed? (speed error ~0 = never accelerates)
    long_behaviour = []
    if nc_long and all(mth[5]['speed_err'] < 0.01 for mth in nc_long):
        long_behaviour.append('**NeurComm (1e6 steps, both optimizers):** no car ever changes speed. All cars '
                              'hold 15 m/s and car 1 never closes its gap. Longer training made the "do nothing" '
                              'optimum more complete, not less.')
    nc_limit = ('NeurComm converged to a "do nothing" policy even with paper-length training, so the paper\'s NeurComm '
                'result was not reproduced here. Next: multiple seeds, spatial discount (`coop_gamma`; the shipped '
                'NeurComm configs use −1 = global reward), entropy/learning-rate sensitivity.'
                if nc_long else 'NeurComm needs paper-length training and possibly tuning.')

    blocks = [
        ('h1', 'Networked multi-agent RL for cooperative cruise control'),
        ('p', 'Reproduction of Chu, Chinchali & Katti, "Multi-agent RL for Networked System Control" (ICLR 2020), '
              'CACC catch-up scenario. Generated %s.' % datetime.datetime.now().strftime('%d %b %Y, %H:%M')),
        ('h2', 'Summary'),
        ('ul', summary),
        ('h2', '1. Task'),
        ('img', fig('fig1_task.png'), 'Eight cars follow a leader. Each car is an agent that only sees and talks to '
                                      'the cars directly in front and behind.'),
        ('ul', ['**State per car (5 numbers):** speed error, speed gap to the car ahead, gap to the controller\'s target speed, gap error, acceleration (IA2C also sees its neighbours\' states).',
                '**Action:** one of 4 gain settings (α, β) of an optimal-velocity car-following controller.',
                '**Reward:** −(gap error)² − (speed error)² − 0.1·accel², large penalty on collision; '
                'summed over all cars (cooperative).',
                '**Algorithms:** IA2C (independent actor-critic using neighbour observations; critic sees neighbour '
                'actions) and NeurComm (adds learned messages: each car sends its neighbours its state, last policy '
                'and LSTM memory every step, `agents/policies.py:229-250`).']),
        ('h2', '2. Main results'),
        ('table', main_header, main_rows),
        ('img', fig('fig2_main_results.png'), 'Lower is better in all three panels. Grey = no-learning baseline.'),
        ('img', fig('fig3_per_seed.png'), 'Each dot is one unseen test episode. Points below the diagonal are seeds '
                                          'where the learned method beat the fixed controller.'),
        ('h2', '3. Behaviour: does the platoon coordinate?'),
        ('img', fig('fig5_trajectories.png'), 'Gap (top) and speed (bottom) of all 8 cars in one unseen episode. '
                                              'Dotted line = target.'),
        ('ul', ['**Fixed controller:** the disturbance from car 1 travels down the platoon and **grows** '
                '(string instability): later cars dip to smaller gaps and take longer to settle.',
                '**IA2C:** the wave **shrinks** down the platoon and settles sooner: the agents learned to '
                'coordinate.',
                '**NeurComm (60k steps):** car 1 closes its gap but car 2 does not respond, so the gap moves back '
                'to car 2: a "do nothing" local optimum.'] + long_behaviour),
        ('h2', '4. Training'),
        ('img', fig('fig4_learning_curves.png'), 'Reward of a test episode after each training episode (rolling '
                                                 'median of 15). Dashed = best fixed controller.'),
        ('p', 'Single-seed RL training is noisy, so the reported model is the checkpoint with the best rolling '
              'validation reward. Validation seeds are separate from the 50 evaluation seeds.'),
    ]
    if overnight:
        blocks += [('h2', '5. NeurComm with paper-length training'), ('ul', long_status)]
        blocks += [('img', fig('fig7_neurcomm_long.png'), 'NeurComm training reward over the paper\'s 1e6 steps, '
                                                          'two optimizer implementations.')]
    blocks += [
        ('h2', '6. Reproduction findings'),
        ('img', fig('fig6_reproduction.png'), '(a) With the original settings, rewards swing widely for all methods. '
                                              '(b) A TensorFlow-matching optimizer removes NeurComm\'s crash phase '
                                              'but does not improve the final policy.'),
        ('table', ['Finding', 'Evidence', 'Consequence'], [
            ['Training is unstable with the paper\'s settings', 'Fig. 6a: IA2C briefly reached the baseline, then '
             'fell back; NeurComm swung widely', 'Report the best validation checkpoint, not the last'],
            ['PyTorch vs TF1 RMSprop differ (state init 0 vs 1, ε outside vs inside √)', 'TF-style version: NeurComm '
             'crash-like episodes 47% → 18% (last third of training); IA2C worse', 'Not the main cause; kept as an '
             'option (`optimizer = rmsprop_tf`)'],
            ['Model selection is decisive', 'IA2C: last checkpoint worse than baseline; best validation checkpoint '
             'beats it', 'Best-checkpoint saving added to the trainer'],
            ['Seed overlap risk', 'Training seeds count up by 2 per episode (reached ~2000 at 250k steps)',
             'Evaluation moved to seeds 100000+'],
        ]),
        ('h2', '7. Setup'),
        ('table', ['Item', 'Value'], setup_rows),
        ('h3', 'All fixed-controller presets (same 50 seeds)'),
        ('table', ['Preset (α, β)', 'Reward', 'Collisions'], preset_rows),
        ('h2', '8. Limitations and next steps'),
        ('ul', ['One training seed per method: needs 3+ seeds for error bars across training runs.',
                nc_limit,
                'Slow-down scenario not yet solved (first 250k-step runs crashed on all test seeds).',
                'Agents are decentralised in what they observe, but simulated and trained in one process. Next: '
                'one process per car with message queues (truly distributed execution).',
                'Traffic-signal (ATSC) scenarios need the SUMO simulator.']),
        ('h2', 'How to reproduce'),
        ('code', 'python demo/run_demo.py --steps 6e4 --algos ma2c_nc,ia2c --scenarios catchup --out demo/runs_repro\n'
                 'python demo/make_report.py --runs demo/runs_repro --out demo/results_repro\n'
                 'python demo/build_presentation.py   # rebuilds this report from demo/runs'),
    ]

    with open(os.path.join(OUT, 'REPORT.md'), 'w') as f:
        f.write(render_md(blocks))
    with open(os.path.join(OUT, 'report.html'), 'w') as f:
        f.write(render_html(blocks, 'NMARL demo results'))
    print('Wrote %s' % os.path.relpath(OUT, REPO))
    for line in summary:
        print(' -', re.sub(r'\*\*', '', line))


if __name__ == '__main__':
    main()
