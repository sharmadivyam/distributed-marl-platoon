#!/usr/bin/env bash
# Demo runs of the MPI platoon. Run from the repository root:  bash mpi/run.sh
# Each run starts 9 MPI processes (8 cars + the road) and writes a log the dashboard can replay.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MPI="mpirun --oversubscribe -np 9 $PY mpi/platoon_mpi.py"
mkdir -p mpi/logs

echo "1) No faults: MPI gives exactly the report's numbers"
$MPI --controller ia2c  --log mpi/logs/demo_ia2c.jsonl
$MPI --controller fixed --log mpi/logs/demo_fixed.jsonl

echo "2) Messages 0.5 s late: the fixed controller crashes, unless it falls back to its radar"
$MPI --controller fixed --delay 5 --fallback last  --log mpi/logs/demo_delay_last.jsonl
$MPI --controller fixed --delay 5 --fallback radar --log mpi/logs/demo_delay_radar.jsonl

echo "3) 20% of messages lost"
$MPI --controller ia2c --drop 0.2 --log mpi/logs/demo_ia2c_drop20.jsonl

echo "4) Car 4's process crashes at 25 s, then the leader brakes at 30 s"
$MPI --controller fixed --kill-car 4 --kill-at 25 --brake-at 30 --crash-timeout-ms 500 --log mpi/logs/demo_crash_brake.jsonl

echo
echo "Watch them in the browser (four runs side by side):"
echo "  $PY -m simulator.server --port 8766 --open --replay mpi/logs/demo_delay_last.jsonl \\"
echo "      mpi/logs/demo_delay_radar.jsonl mpi/logs/demo_crash_brake.jsonl mpi/logs/demo_ia2c_drop20.jsonl"
