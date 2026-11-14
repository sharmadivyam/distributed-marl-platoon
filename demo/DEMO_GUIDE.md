# Demo guide: showing networked multi-agent RL working

How to show your professor that the implementation runs, learns, and beats a hand-tuned
controller, plus an honest account of where NeurComm stands. Everything runs locally on
CPU. No SUMO is needed: the demo uses the CACC (platoon) scenario.

## For the meeting: open the presentation

- **`demo/presentation/report.html`**: double-click to open in a browser. It works offline,
  with all figures embedded. Print to PDF from the browser if needed (File → Print → Save as
  PDF).
- **`demo/presentation/REPORT.md`**: the same report. In VS Code, open it and press Cmd+Shift+V.
- **`demo/presentation/fig*.png`**: individual figures for slides.
- **`demo/presentation/tables/*.csv`**: the numbers behind the tables.

| Figure | Shows |
|---|---|
| `fig1_task.png` | The task: 8 cars, neighbour-only links |
| `fig2_main_results.png` | Penalty, gap error and speed error per method, with 95% CIs |
| `fig3_per_seed.png` | Paired per-seed comparison against the fixed controller |
| `fig4_learning_curves.png` | Training curves |
| `fig5_trajectories.png` | Gap and speed of every car in one test episode |
| `fig6_reproduction.png` | The instability and the optimizer A/B (reproduction findings) |
| `fig7_neurcomm_long.png` | NeurComm paper-length training (updates when rebuilt) |

**Tomorrow morning, before the meeting:** rebuild so the overnight NeurComm runs are
included (≈1 min):
```bash
cat demo/runs_overnight/*.out        # "All runs done" = finished and evaluated
python demo/build_presentation.py    # adds NeurComm 1e6 rows to the table if evaluated
```
If they're still running, the report shows their progress so far. You don't need to wait.

## Results at a glance (CACC catch-up, 50 unseen test seeds)

| Method | Reward (higher is better) | Collisions | Headway err [m] | Speed err [m/s] |
|---|---|---|---|---|
| Fixed controller, best preset (no learning) | −72.7 ± 46.5 | 0 / 50 | 0.88 | 0.75 |
| **IA2C** (best checkpoint @ 27k steps) | **−37.2 ± 23.9** | 0 / 50 | **0.53** | **0.33** |
| NeurComm (best checkpoint @ 8.8k steps) | −422.3 ± 270.4 | 0 / 50 | 2.53 | 0.11 |
| NeurComm, 1e6 steps, PyTorch RMSprop (best @ 470k) | −392.4 ± 245.0 | 0 / 50 | 2.36 | 0.00 |
| NeurComm, 1e6 steps, TF-style RMSprop (best @ 396k) | −392.4 ± 245.0 | 0 / 50 | 2.36 | 0.00 |

- **IA2C learned a better controller than the hand-tuned one:** about half the penalty, with
  lower gap and speed errors and no crashes.
- **NeurComm did not reproduce**, even at the paper's 1e6 steps with either optimizer. It
  converged to a "do nothing" policy: every car holds 15 m/s, car 1 never closes its gap,
  and the score equals the fixed (0, 0) preset exactly. This is an honest reproduction gap
  (see "What we found").

Figures (in `demo/results/`):
- `trajectories_catchup.png`: the fixed controller's gap wave **grows** down the platoon
  (car 8 dips to ~12 m) and needs ~35 s to settle. IA2C's wave **shrinks** and settles by
  ~20 s. NeurComm: car 1 closes its gap but car 2 never reacts, so the gap moves back to
  car 2.
- `learning_curves_catchup.png`: reward during training. Note how noisy single-seed RL
  training is.

## The task in one paragraph

8 cars drive in a line behind a leader. Each car is an agent that sees only itself and the
cars directly in front and behind. Every 0.1 s, each car picks one of 4 gain settings
(α, β) for its car-following controller. Goal: hold a **20 m gap** at **15 m/s** smoothly,
without crashing. In **catch-up**, the front car starts 1.5–2.5× too far back and the whole
platoon has to adjust.

---

## Step 0: one-time setup (5 min)

```bash
cd ~/projects/marl/deeprl_network
python3 -m venv .venv            # skip if .venv already exists
source .venv/bin/activate
pip install -r demo/requirements.txt
python -c "import torch, numpy, pandas, matplotlib; print('ok')"
```

## Step 1: quick sanity run (≈5–10 min)

```bash
python demo/run_demo.py --steps 2e4 --out demo/runs_quick
python demo/make_report.py --runs demo/runs_quick --out demo/results_quick
```
Expected: `All runs done.`, then a table. The results will be poor because there's too
little training. This step only checks that everything runs.

## Step 2: reproduce the demo results (≈20 min)

```bash
python demo/run_demo.py --steps 6e4 --algos ma2c_nc,ia2c --scenarios catchup --out demo/runs_repro
python demo/make_report.py --runs demo/runs_repro --out demo/results_repro
```
- Trains IA2C and NeurComm in parallel, keeps each one's **best checkpoint**, and evaluates
  it on 50 unseen seeds.
- With the same seeds, training was deterministic in our tests: the overnight runs exactly
  matched the 60k runs over their first 60k steps. Expect the same or very close numbers.
- The run that produced the table above is already saved in `demo/runs/`, with its report in
  `demo/results/`.

Watch training live (good to show in the meeting):
```bash
tensorboard --logdir demo/runs_repro      # open http://localhost:6006 → train_reward
```

## Step 3: rebuild the report from saved runs (≈1 min)

```bash
python demo/make_report.py          # reads demo/runs, writes demo/results
```

| File | What it shows |
|---|---|
| `summary.md` | Table: reward, collisions, gap error, speed error, and which checkpoint was evaluated |
| `learning_curves_catchup.png` | Reward vs training step. Dashed line = best fixed controller |
| `trajectories_catchup.png` | Gap and speed of all 8 cars in one unseen test episode, per method |

## Step 4: meeting flow (≈10 min)

1. **Problem (1 min):** 8 cars, neighbour-only information, no central controller.
2. **Method (2 min):** A2C agents. Coordination comes from neighbour observations (IA2C) or
   learned messages (NeurComm). Show the code:
   - NeurComm message construction: `agents/policies.py:229-250`
   - Spatial-discount return: `agents/utils.py:157-173`
3. **It runs (2 min):** start Step 1 live and show TensorBoard.
4. **Results (3 min):**
   - The table: IA2C beats the hand-tuned controller.
   - `trajectories_catchup.png`: the smaller, shrinking gap wave *is* coordination.
5. **Honest status + next steps (2 min):**
   - NeurComm did not reproduce even at paper length: it converges to "do nothing" (Fig. 7).
   - Reproduction findings: see "What we found" below.
   - Next: one process per car, multiple seeds.

## Step 5: NeurComm paper-length runs (finished)

Two runs at 1e6 steps, identical except for the optimizer implementation:
- `demo/runs_overnight/torch/`: original PyTorch RMSprop.
- `demo/runs_overnight/tf/`: TF1-style RMSprop.

Check progress:
```bash
grep -o "global step [0-9]*" demo/runs_overnight/*/ma2c_nc_catchup/log/*.log | tail -2
cat demo/runs_overnight/*.out          # says "All runs done" when finished
```
When done:
```bash
python demo/make_report.py --runs demo/runs_overnight/torch --out demo/results_overnight_torch
python demo/make_report.py --runs demo/runs_overnight/tf    --out demo/results_overnight_tf
```
Result (7 Oct 2026): both finished. Their best checkpoints scored −392.4, worse than the
fixed controller's −72.7, so this is reported as a reproduction finding. `build_presentation.py`
already includes them in the table and in Fig. 7.

To restart them (e.g. after a reboot), delete the folder and run:
```bash
caffeinate -i python demo/run_demo.py --steps 1e6 --algos ma2c_nc --scenarios catchup \
    --optimizer rmsprop --out demo/runs_overnight/torch
```

## What we found while reproducing (worth telling the professor)

1. **Training is unstable with the original settings.** In 250k-step runs
   (`demo/runs_v1_250k/`), IA2C briefly matched the hand-tuned controller and then fell away,
   and NeurComm swung wildly before settling into a "do nothing" policy. These were the
   authors' own PyTorch branch and the paper's hyperparameters.
2. **Optimizer port difference.** The paper's code used TensorFlow 1's RMSProp, which starts
   its running average at 1 and puts ε inside the square root. PyTorch starts at 0 and puts ε
   outside, so it takes larger steps. A TF-matching version made NeurComm's training steadier
   (crash-like episodes in the last third of training: 47% → 18%) but did not improve the final policy, and it made IA2C
   worse. So it's not the main cause.
3. **Model selection matters most.** Keeping the checkpoint with the best rolling validation
   reward, instead of the last one, turned IA2C from "worse than baseline" into "2× better
   than baseline".
4. **NeurComm does not reproduce at paper length.** Two 1e6-step runs (one per optimizer)
   both converged to "do nothing" (score exactly equal to the fixed (0, 0) preset; speed
   error 0.00).
5. **Seed hygiene.** The training/validation seeds count upward (12, 14, …) and reached about
   1996 in a 250k run. That's why evaluation uses seeds 100000–100490.

## How to read the outputs

- **Reward** = average per-step platoon reward. It is always ≤ 0, and closer to 0 is better.
  A crash costs −1000 per car per step for the rest of the episode.
- **Collisions** = test episodes where any gap fell below 1 m.
- **Gap / speed error** = average distance from 20 m / 15 m/s over all cars and steps.
- **Best checkpoint** = the training checkpoint with the highest average over 5 validation
  episodes. Validation seeds never overlap the 50 evaluation seeds, so the comparison is
  fair.

## Likely questions and short answers

| Question | Answer |
|---|---|
| Is this truly distributed? | Decisions use only neighbour information, but all agents are simulated in one process and trained jointly. One process per car is the next step. |
| Isn't picking the best checkpoint cheating? | No. Selection uses separate validation seeds. All numbers in the table come from 50 different, unseen seeds. |
| Why does IA2C beat the fixed controller? | It learned a per-car combination of settings: cars 1, 3, 5, 6 use (0.5, 0.5) and cars 2, 4, 7, 8 use (0, 0.5). At test time each car essentially keeps its setting. This mix damps the gap wave down the platoon, which no single uniform setting does. (A search over all 4^8 fixed per-car combinations would be a stronger baseline: a fair next step.) |
| Why didn't NeurComm work? | Even at the paper's 1e6 steps it settles in a safe "do nothing" policy: never accelerating never crashes, but never fixes the gap. Candidate causes to test: one training seed, the shipped NeurComm configs use `coop_gamma = -1` (global reward, no spatial discount), and differences from the paper's TensorFlow code. |
| How solid is this statistically? | One training seed per method and 50 test seeds. Error bars across training seeds would need 3+ runs per method. |
| Why not the traffic-light scenarios? | They need the SUMO simulator and much longer training. The code supports them. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: traci` | You chose an ATSC scenario. The demo uses `--scenarios catchup` (or `slowdown`). |
| A run is skipped | It already has a checkpoint. Delete its folder to retrain. |
| A run finished with code 1 | Read `<run>/train_stdout.log`. |
| No curve in TensorBoard yet | The first point appears after the first episode (~1 min). |
| "No runs found" | Pass `--runs` with the folder you trained into. |
| Overnight run stopped | The Mac slept or rebooted. See the restart command in Step 5. |

## What was changed in the original code

| File | Change | Why |
|---|---|---|
| `main.py` | ATSC environments imported only when needed | CACC runs without SUMO |
| `main.py` | `evaluate --checkpoint best\|last` (default `last`) | Evaluate the best checkpoint |
| `utils.py` | Trainer saves the best rolling-validation checkpoint to `model/best/` | Model selection |
| `utils.py`, `gather_data.py` | `sudo cp` → `shutil.copy` | No password prompt |
| `agents/utils.py` | `np.bool` → `bool`; new `TFRMSprop` | NumPy compatibility; optimizer experiment |
| `agents/models.py` | Optional `optimizer = rmsprop_tf` config key (default unchanged) | Optimizer experiment |
| `demo/` | `run_demo.py`, `make_report.py`, `requirements.txt`, this guide | The demo pipeline |

No changes to the algorithms, network architecture, hyperparameters or reward.
