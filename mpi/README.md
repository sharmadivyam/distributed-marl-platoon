# The platoon on MPI

Every car runs as its **own MPI process**. Cars talk to each other only through MPI messages
between neighbouring processes, like real vehicle-to-vehicle (V2V) radio links.

```
rank 0  = the road: physics + the leading vehicle
rank 1  = car 1  ─┐
rank 2  = car 2   │  each car: gets its own sensor readings from the road,
  ...             │  exchanges V2V messages with its neighbour processes,
rank 8  = car 8  ─┘  runs ITS OWN trained policy, sends back an acceleration command
```

Three controllers:

| `--controller` | What each car process runs | What neighbours send each other every 0.1 s |
|---|---|---|
| `fixed` | hand-tuned OVM setting | only the front car's speed (1 number) |
| `ia2c` | its own trained IA2C network | + its observation (5 numbers) |
| `neurcomm` | its own part of the trained NeurComm network (encoders, LSTM, actor head) | + observation, last policy and LSTM hidden state (5 + 4 + 64 = 73 numbers) |

NeurComm uses our best NeurComm model (spatial discount 0.9, seed 13, in `demo/repro/`); pick
another with `--nc-runs`. On seed 100000 it sends 2.47 MB in 60 s, IA2C 0.19 MB (13x less).

Training stays in Python/PyTorch (as before). MPI runs the **trained** policies, which is how a
learned controller would be deployed on real cars: train once, run distributed.

## Setup (once)

```bash
brew install open-mpi                 # the MPI library (Open MPI 5)
.venv/bin/pip install mpi4py          # Python bindings for MPI
```

## Run

```bash
# 9 processes: 8 cars + the road
mpirun --oversubscribe -np 9 .venv/bin/python mpi/platoon_mpi.py --controller ia2c
mpirun --oversubscribe -np 9 .venv/bin/python mpi/platoon_mpi.py --controller fixed
mpirun --oversubscribe -np 9 .venv/bin/python mpi/platoon_mpi.py --controller neurcomm

bash mpi/run.sh                       # the standard demo runs (writes mpi/logs/demo_*.jsonl)
python mpi/check_mpi.py               # proves MPI = Python, step by step
python mpi/experiments.py             # loss/delay sweeps + scaling, graphs in mpi/results/
```

Watch MPI runs in the browser. The demo page has sections (normal driving, leader brakes, late
messages, lost messages, crashed car, malicious car); pick one at the top left and Play plays
only that section, with every controller side by side:

```bash
.venv/bin/python mpi/make_demo_sections.py            # records the 26 runs (a few minutes)
.venv/bin/python -m simulator.server --port 8767 --open --sections mpi/demo_sections.json
```

Any single log can also be replayed: `--replay mpi/logs/<run>.jsonl [more logs]`.

## Options

| Flag | Meaning |
|---|---|
| `--controller fixed\|ia2c` | hand-tuned controller, or the trained IA2C policy (one network per car process) |
| `--drop 0.2` | each V2V message is lost with probability 20% |
| `--delay 5` | V2V messages arrive 5 steps (0.5 s) late |
| `--kill-car 4 --kill-at 25` | car 4's process goes silent at 25 s (crash-stop) |
| `--byzantine-car 4 --byzantine-at 25` | car 4 keeps driving but lies in its messages from 25 s |
| `--attack speed\|noise`, `--lie 10` | claim a false speed (+10 m/s) or send random values |
| `--defense` | each car checks the front car's claimed speed against its own radar |
| `--gossip` | with `--defense`: the car behind a liar warns the car in front of it (2-hop link) |
| `--brake-at 30` | the leader brakes at 30 s |
| `--fallback last\|radar` | when a message is missing: reuse the last value, or estimate the front car's speed from the car's own radar |
| `--cars 16` | bigger platoon (fixed controller only); use `-np 17` |
| `--timeout-ms 50` | how long a car waits for a neighbour's message before using its fallback |
| `--realtime` | run at real speed (10 steps per second), to watch the processes in Activity Monitor |
| `--log file.jsonl` | log for the dashboard and the experiments |

## How one 0.1 s step works

1. The road sends each car its own sensor readings (speed, acceleration, radar gap).
2. **V2V round 1:** each car sends its speed to the car behind (`isend`). The leader's speed comes
   from the road process.
3. Each car waits (up to the timeout) for the front car's message (`iprobe` + `recv`). If it is
   missing or late, it uses its fallback and counts a "stale read".
4. **V2V round 2 (IA2C only):** each car shares its 5-number observation with both neighbours.
5. Each car runs its own policy, turns the chosen setting into an acceleration, and sends it to
   the road.
6. The road waits for every car's command. If a car does not answer within `--crash-timeout-ms`,
   the road marks it as crashed, and from then on that car just rolls on at constant speed.
7. The road moves all cars and writes the step to the log.

## Results (seed 100000; graphs and CSVs in `results/`)

**Correctness:** with no faults, MPI = Python at all 600 steps (IA2C −22.96, fixed −44.51,
NeurComm −60.25).

**Message loss** (`results/drop_sweep.png`), avg reward per step:

| Lost | Fixed, last value | Fixed, radar | IA2C, last value | IA2C, radar |
|---|---|---|---|---|
| 0% | −44.5 | −44.5 | −23.0 | −23.0 |
| 20% | −47.3 | −45.0 | −23.3 | −23.0 |
| 50% | −60.6 | −46.1 | −24.5 | −23.1 |

**Message delay** (`results/delay_sweep.png`):

| Delay | Fixed, last value | Fixed, radar | IA2C, last value | IA2C, radar |
|---|---|---|---|---|
| 0 s | −44.5 | −44.5 | −23.0 | −23.0 |
| 0.3 s | −107.5 | −47.3 | −26.8 | −23.3 |
| 0.5 s | **collision at 18.7 s** | −47.3 | −29.8 | −23.3 |
| 1.0 s | **collision at 15.9 s** | −47.3 | −38.8 | −23.3 |

**Same sweeps on 10 test seeds** (100000 to 100090; `python mpi/experiments.py --test-seeds 10`,
graphs `results/{drop,delay}_sweep_10seeds.png`), mean ± std over the 10 seeds, and how many of
the 10 runs ended in a collision. The ± is large because some roads are harder than others; every
seed moves the same way.

| Lost | Fixed, last value | Fixed, radar | IA2C, last value | IA2C, radar |
|---|---|---|---|---|
| 0% | −56.6 ± 35.0 | −56.6 ± 35.0 | −29.0 ± 17.6 | −29.0 ± 17.6 |
| 20% | −60.5 ± 37.6 | −57.3 ± 35.4 | −29.4 ± 17.8 | −29.1 ± 17.6 |
| 50% | −76.1 ± 48.4 | −58.4 ± 36.1 | −31.1 ± 18.7 | −29.2 ± 17.7 |

| Delay | Fixed, last value | Fixed, radar | IA2C, last value | IA2C, radar |
|---|---|---|---|---|
| 0 s | −56.6, 0/10 crashed | −56.6, 0/10 | −29.0, 0/10 | −29.0, 0/10 |
| 0.3 s | −180.5, **2/10 crashed** | −60.0, 0/10 | −33.5, 0/10 | −29.4, 0/10 |
| 0.5 s | −341.8, **7/10 crashed** | −60.0, 0/10 | −37.1, 0/10 | −29.4, 0/10 |
| 1.0 s | −488.4, **10/10 crashed** | −60.0, 0/10 | −47.8, 0/10 | −29.4, 0/10 |

No collisions at all under message loss, for any controller.

**NeurComm under faults (10 seeds):** with 1 s delay and "last value", NeurComm drops from
−75.8 to −199.7 and crashes on 1/10 roads; IA2C goes from −29.0 to −47.8 with no crash. With the
radar fallback IA2C stays at −29.4, NeurComm only recovers to −83.2: the radar can replace the
front car's speed, but not a neighbour's old hidden state. Loss barely matters for any controller.

**Malicious (Byzantine) car** (seed 100000, car 4 lies from 25 s, leader brakes at 30 s;
normal brake scores: fixed −147.7, IA2C −160.0, NeurComm −191.7):

| Attack | Fixed | IA2C | NeurComm |
|---|---|---|---|
| claims +10 m/s | **collision 39.1 s** | −215.1 | −189.7 |
| claims +10 m/s, radar check | −147.7 | −160.0 | −197.8 |
| random values | **collision 38.5 s** | **collision 41.8 s** | −189.3 |
| random values, radar check | −147.5 | −159.9 | −189.6 |

The radar check (`--defense`): a car compares its front car's claimed speed with the speed its
radar implies (how fast the gap changes) and ignores the message when they differ by more than
2 m/s. No message was rejected in runs without a liar (no false alarms). The car in front of the
liar cannot check it (the radar only looks forward), which is why NeurComm, whose messages also go
forward, is not fully protected.

**Malicious car on 10 seeds** (`python mpi/experiments.py --byzantine 10`, graph
`results/byzantine_10seeds.png`): car 4 lies from 25 s, the leader brakes at 30 s.

| What car 4 sends | Fixed, no defence | IA2C, no defence | NeurComm, no defence | Lies caught by the radar check |
|---|---|---|---|---|
| nothing false | −159.8, 0/10 crashed | −166.1, 0/10 | −207.2, 0/10 | 0 false alarms in 30 runs |
| +1.5 m/s | −161.9, 0/10 | −168.1, 0/10 | −212.3, 0/10 | 0% (below the 2 m/s threshold) |
| +5 m/s | −172.3, 0/10 | −178.5, 0/10 | −225.5, 0/10 | 100% |
| +10 m/s | **10/10 crashed** | −221.2, 0/10 | −204.4, 0/10 | 100% |
| random values | **10/10 crashed** | **3/10 crashed** | −204.1, 0/10 | 86% (the rest were close to the truth) |

With the radar check on, no run crashed and Fixed and IA2C got back exactly their no-liar
scores. Small lies slip through the check but also do little harm; big lies are always caught.
NeurComm did not crash under any lie (we do not fully know why yet; it reacts much more to late
messages than to false ones).

**Gossip** (`--defense --gossip`): the car behind the liar is the only one whose radar can check
it. After 3 rejected messages in a row it warns the car in front of the liar directly over a
2-hop link (around the liar), and both then ignore the liar. Witness rule: a car only believes an
accusation about its back neighbour from the car right behind that neighbour. On seed 100000 the
liar is exposed 0.2 s after it starts lying, and NeurComm goes from −197.8 (radar check only) to
−191.1 (no-liar score −191.7). Cost: one tiny gossip message per car per step. Limit: the liar is
itself the witness for the car in front of it, so it could falsely accuse that car; stopping this
needs more witnesses or signed messages.

**Crashed car process:** car 4 goes silent at 25 s, the leader brakes at 30 s: collision at
37.0 s (fixed) and 37.9 s (IA2C). Without the crash: no collision.

**Scaling** (`results/scaling.png`, fixed controller, best of 3 runs of 30 s):

| Cars | MPI processes | Time per 0.1 s step | V2V messages per step |
|---|---|---|---|
| 8 | 9 | 0.8 ms | 8 |
| 16 | 17 | 1.3 ms | 16 |
| 32 | 33 | 2.1 ms | 32 |

The full 60 s runs with 16 and 32 cars end in collisions (44.1 s and 33.8 s), exactly as in plain
Python. The hand-tuned controller lets the disturbance grow down a longer platoon
(string instability). That is physics, not MPI.

**Note on IA2C:** the MPI runs use the seed-12 IA2C policy, the best of the three we trained
(see the reproduction check in `demo/repro/`; seeds 13 and 14 scored −94.5 and −374.6).

## Talking points

- **Same algorithm, now distributed:** the MPI run reproduces the Python results exactly, at every
  one of the 600 steps (`check_mpi.py`).
- **Delay is worse than loss:** losing 20% of messages barely matters, but 0.5 s old information
  makes the hand-tuned platoon crash.
- **Local sensing is the best fallback:** estimating the front car's speed from the radar,
  instead of trusting an old message, removes that crash.
- **Node failure:** standard MPI aborts the whole job when a process dies, so we model a crash as a
  process that goes silent (crash-stop). The road detects it with a timeout. A crashed car cannot
  brake, so when the leader brakes the platoon collides: detection alone is not enough, the car
  itself needs a safe fallback.
- **Learning helped robustness here:** IA2C stays safe under every delay we tried, while the
  hand-tuned controller crashes from 0.5 s on (unless it uses the radar fallback).
- **Scales cheaply:** time per step grows about linearly with the number of processes and stays
  far below the 100 ms real-time budget.
- **Timeouts under load:** with a short timeout on a busy laptop, a slow message can look like a
  lost one. That is the classic false-suspicion problem of failure detectors.

## Files

| File | Role |
|---|---|
| `platoon_mpi.py` | the MPI program (road rank + car ranks, fault injection, logging) |
| `check_mpi.py` | MPI vs Python, step by step |
| `experiments.py` | loss and delay sweeps, scaling test, CSV + graphs in `results/` |
| `run.sh` | the standard demo runs |
| `make_demo_sections.py` | records the runs for the sectioned demo page (`demo_sections.json`) |
| `train_mpi.py`, `check_train.py`, `compare_training.py` | distributed training, its exactness check, comparison |
