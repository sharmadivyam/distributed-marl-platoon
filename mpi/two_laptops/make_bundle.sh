#!/usr/bin/env bash
# Run on laptop A (from anywhere). Packs the code + trained policies into ~/Downloads/platoon_bundle.zip
# for laptop B (send it with AirDrop). Excludes the virtualenv, git history and experiment outputs.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TMP="$(mktemp -d)/platoon"
mkdir -p "$TMP"
rsync -a --exclude '.venv' --exclude '.git' --exclude 'demo/repro' --exclude 'demo/runs_*' \
      --exclude 'demo/videos' --exclude 'demo/presentation*' --exclude 'demo/results*' \
      --exclude 'mpi/logs' --exclude 'mpi/results' --exclude 'envs/*_data' --exclude '__pycache__' \
      "$REPO/" "$TMP/"
(cd "$(dirname "$TMP")" && rm -f ~/Downloads/platoon_bundle.zip && zip -qr ~/Downloads/platoon_bundle.zip platoon)
echo "Bundle: ~/Downloads/platoon_bundle.zip ($(du -h ~/Downloads/platoon_bundle.zip | cut -f1)). AirDrop it to laptop B."
