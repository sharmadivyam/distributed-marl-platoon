#!/usr/bin/env bash
# Run on laptop A, inside /Users/Shared/platoon:
#   bash mpi/two_laptops/run_two_laptops.sh                 # quick check: which machine runs which rank
#   bash mpi/two_laptops/run_two_laptops.sh --controller ia2c --log mpi/logs/two_laptops_ia2c.jsonl
set -e
cd /Users/Shared/platoon
HOSTS=mpi/two_laptops/hosts
[ -f "$HOSTS" ] || { echo "Create $HOSTS from hosts.example first"; exit 1; }
mkdir -p mpi/logs
if [ $# -eq 0 ]; then
  echo "== 1) Can MPI start processes on both laptops?"
  mpirun --hostfile "$HOSTS" -np 9 --map-by slot hostname | sort | uniq -c
  exit 0
fi
echo "== Platoon across two laptops (road + cars 1-4 on A, cars 5-8 on B)"
mpirun --hostfile "$HOSTS" -np 9 --map-by slot \
       /Users/Shared/platoon/.venv/bin/python /Users/Shared/platoon/mpi/platoon_mpi.py "$@"
