#!/usr/bin/env bash
# Reproduce the whole local environment from a clean checkout.
# Run from the repo root under Git Bash:   bash scripts/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

clone() {  # clone <url> <dir>
  if [ -d "$2/.git" ]; then
    echo "  $2 already present"
  else
    git clone --depth 1 -q "$1" "$2"
    echo "  cloned $2"
  fi
}

echo "== datasets =="
mkdir -p data
clone https://github.com/arcprize/ARC-AGI-2.git              data/ARC-AGI-2
clone https://github.com/arcprize/ARC-AGI.git                data/ARC-AGI-1
clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter.git data/ARC-AGI-3-Kaggle-Starter

echo "== agi2 sanity check =="
python agi2/run_baseline.py --split evaluation | head -4

echo "== agi3 environment (needs Python 3.12; uv installs it) =="
STARTER=data/ARC-AGI-3-Kaggle-Starter
if [ ! -d "$STARTER/.venv" ]; then
  uv python install 3.12
  uv venv --python 3.12 "$STARTER/.venv"
fi

# Windows venvs put binaries in Scripts/, POSIX ones in bin/.
if [ -x "$STARTER/.venv/Scripts/python.exe" ]; then
  VPY="$STARTER/.venv/Scripts/python.exe"
else
  VPY="$STARTER/.venv/bin/python"
fi

uv pip install --python "$VPY" -q pip "arc-agi>=0.9.6" "kaggle>=2.2" \
  python-dotenv pandas pyarrow

clone https://github.com/arcprize/ARC-AGI-3-Agents.git "$STARTER/vendor/ARC-AGI-3-Agents"

# Upstream bug: slim_framework.py writes an em-dash with the platform default
# encoding, which produces an undecodable agents/__init__.py on a GBK/CP1252
# Windows box. Force UTF-8.
if grep -q 'INIT.write_text(SLIM)$' "$STARTER/scripts/slim_framework.py"; then
  sed -i 's/INIT.write_text(SLIM)$/INIT.write_text(SLIM, encoding="utf-8")/' \
    "$STARTER/scripts/slim_framework.py"
  echo "  patched slim_framework.py for UTF-8"
fi
"$VPY" "$STARTER/scripts/slim_framework.py"

echo
echo "Done. Next:"
echo "  python agi2/run_baseline.py --split training"
echo "  python agi2/inspect_tasks.py --split evaluation --limit 12 --predict"
echo "  (cd $STARTER && .venv/Scripts/python.exe scripts/play_local.py --game ls20 --max-steps 50)"
