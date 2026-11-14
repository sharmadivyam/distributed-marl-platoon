# Speaker notes: final presentation (about 10 minutes)

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
.venv/bin/python -m simulator.server --port 8766 --open --replay mpi/logs/demo_delay_last.jsonl \
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
