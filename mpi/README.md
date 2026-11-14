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

bash mpi/run.sh                       # the standard demo runs (writes mpi/logs/demo_*.jsonl)
python mpi/check_mpi.py               # proves MPI = Python, step by step
python mpi/experiments.py             # loss/delay sweeps + scaling, graphs in mpi/results/
```

Watch MPI runs in the browser (several runs side by side):

```bash
.venv/bin/python -m simulator.server --port 8766 --open --replay \
    mpi/logs/demo_delay_last.jsonl mpi/logs/demo_delay_radar.jsonl \
    mpi/logs/demo_crash_brake.jsonl mpi/logs/demo_ia2c_drop20.jsonl
```

## Options

| Flag | Meaning |
|---|---|
| `--controller fixed\|ia2c` | hand-tuned controller, or the trained IA2C policy (one network per car process) |
| `--drop 0.2` | each V2V message is lost with probability 20% |
| `--delay 5` | V2V messages arrive 5 steps (0.5 s) late |
| `--kill-car 4 --kill-at 25` | car 4's process goes silent at 25 s (crash-stop) |
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

**Correctness:** with no faults, MPI = Python at all 600 steps (IA2C −22.96, fixed −44.51).

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
