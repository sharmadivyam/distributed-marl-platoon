# Speaker notes: 10-minute presentation

## Before the meeting

- Open `demo/presentation_10min/report.html` in the browser (double-click it in Finder).
- Open a terminal in the repo with the backup command (below) typed in, but don't run it.
- Don't train live. 10 minutes is too short, and the report already has the results.

## Timeline: what to show

| Time | Script block | Show in report.html |
|---|---|---|
| 0:00–0:25 | What we did | Title + first paragraph |
| 0:25–1:20 | Problem | Section 1 (8-car figure) |
| 1:20–2:15 | RL from zero | Section 2 (RL loop figure) |
| 2:15–3:00 | A2C | Section 3 (no equations) |
| 3:00–3:45 | IA2C vs NeurComm | Section 4 (A ↔ B ↔ C ↔ D figure) |
| 3:45–4:35 | How we implemented it | Section 5 table |
| 4:35–5:20 | How we tested it | Section 6 + presets table |
| 5:20–6:20 | Results | Section 7: table, key interpretation, trajectory figure |
| 6:20–7:35 | Why NeurComm performed poorly | Section 8 (1e6 table + curve), then section 9 |
| 7:35–8:25 | What it means | Section 9, last bullet |
| 8:25–9:15 | Limitations | Section 10 (only these 3) |
| 9:15–10:00 | Closing | Section 11 |

## Numbers to remember

- Baseline **-72.7** · IA2C **-37.2** · NeurComm **-422.3** · **50** test seeds · **60k** training steps
- IA2C: 0.53 m gap error, 0.33 m/s speed error, 0/50 collisions, better on 50/50 seeds
- NeurComm: 2.53 m gap error, 0.11 m/s speed error, 0/50 collisions
- NeurComm at 1,000,000 steps: -392.4, -392.4

## Wording to watch

- Say **"average reward per step"**, not "cumulative reward". That's what the table reports.
- **Fixed controller** = the non-learning car-following (OVM) controller with one fixed gain preset. It is the baseline.
- **Action** = choosing 1 of 4 gain presets, not acceleration.
- **Hidden state** = the LSTM's internal memory vector, not the physical car state.
- **Reward** is a negative penalty, so closer to zero is better.

## If you get stuck (one sentence)

> We reproduced the CACC setup, trained IA2C and NeurComm, selected the best validation checkpoint, tested on 50 unseen seeds against the best fixed controller, and found that IA2C learned useful coordination while NeurComm converged to a poor do-nothing local optimum.

## Backup: "does the code actually run?" (~20 s)

Evaluates the saved IA2C model on one unseen seed. It works on a **copy**, so the saved results are never touched:

```bash
rm -rf /tmp/ia2c_live && cp -R demo/runs/ia2c_catchup /tmp/ia2c_live
.venv/bin/python main.py --base-dir /tmp/ia2c_live evaluate --evaluation-seeds 100000 --checkpoint best
```

Expected last line: `test 0, avg reward -22.96` (the same seed as the trajectory figure).

## Likely questions

| Question | Short answer |
|---|---|
| Why is the best IA2C checkpoint at 27.1k, not 60k steps? | RL training is noisy; we keep the checkpoint with the best validation reward. Validation seeds never overlap the test seeds. |
| Isn't picking the best checkpoint cheating? | No. It is chosen on separate validation episodes; every number in the table comes from 50 different, unseen seeds. |
| What did IA2C actually learn? | Cars 2, 4, 7, 8 use (0, 0.5); cars 1, 3, 5, 6 use (0.5, 0.5). At test time each car essentially keeps its setting (deterministic policy), so it learned *which car should use which setting*, not switching over time. The trajectory figure shows this mix damps the disturbance down the platoon, which no single uniform setting does. |
| Could you get the same with a fixed per-car combination? | Possibly. Our baseline only tries the same setting for every car (4 options). Searching all 4^8 per-car combinations would be a stronger baseline; that is a fair next step. |
| Why not just train NeurComm longer? | We did: 1,000,000 steps (the paper's length) still converged to do nothing. |
| Is it really distributed? | Each agent uses only neighbour information, but all agents run in one process. One process per car is future work. |
| How sure are you? | One training seed per method, so this is what we observed, not a statistical proof. More seeds are the next step. |
