#!/usr/bin/env bash
# Run on BOTH laptops. Puts the project at /Users/Shared/platoon (the same path on every Mac, so MPI
# can start processes on the other laptop with identical commands) and installs everything it needs.
#   Laptop A:  bash mpi/two_laptops/setup_laptop.sh                (copies this repo)
#   Laptop B:  bash setup_laptop.sh ~/Downloads/platoon_bundle.zip (unzip the bundle first or pass it)
set -e
DEST=/Users/Shared/platoon
SRC="${1:-}"
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
brew list open-mpi >/dev/null 2>&1 || brew install open-mpi

if [ -z "$SRC" ]; then                       # laptop A: copy this repository
  REPO="$(cd "$(dirname "$0")/../.." && pwd)"
  mkdir -p "$DEST"
  rsync -a --exclude '.venv' --exclude '.git' --exclude 'demo/repro' --exclude '__pycache__' "$REPO/" "$DEST/"
else                                         # laptop B: unpack the bundle
  rm -rf /tmp/platoon_unpack && mkdir -p /tmp/platoon_unpack
  unzip -q "$SRC" -d /tmp/platoon_unpack
  mkdir -p "$DEST" && rsync -a /tmp/platoon_unpack/platoon/ "$DEST/"
fi

cd "$DEST"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r demo/requirements.txt mpi4py
# MPI starts remote processes over a non-interactive ssh, which does not read ~/.zshrc:
# make Homebrew (and so Open MPI) visible there too.
# (only needed on laptop B, the one that MPI reaches over ssh)
if [ -n "$SRC" ]; then
  BREW_BIN="$(dirname "$(command -v brew)")"
  grep -q "$BREW_BIN" ~/.zshenv 2>/dev/null || echo "export PATH=$BREW_BIN:\$PATH" >> ~/.zshenv
fi
echo
echo "Done. Project at $DEST"
echo "This laptop's MPI hostname: $(scutil --get LocalHostName).local"
mpirun --version | head -1
