#!/usr/bin/env bash
# cme-bench installer (Linux / macOS / Git-Bash). Usage: ./install.sh [--cuda 118|121|128|cpu] [--python python3.10]
set -euo pipefail
cd "$(dirname "$0")"

CUDA="auto"
PY="${PYTHON:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) CUDA="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

ok_version() { "$@" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3, 10), (3, 11)) else 1)' 2>/dev/null; }
if [[ -z "$PY" ]]; then
  for c in python3.10 python3.11 "py -3.10" "py -3.11" python3 python; do
    if ok_version $c; then PY="$c"; break; fi
  done
fi
if [[ -z "$PY" ]] || ! ok_version $PY; then echo "Python 3.10 or 3.11 is required"; exit 1; fi

if [[ "$CUDA" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo 0)
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    if [[ "${CC:-0}" -ge 12 ]]; then CUDA=128
    elif [[ "${DRV:-0}" -ge 525 ]]; then CUDA=118   # article environment; 121 is optional
    else CUDA=118; fi
  else
    CUDA=cpu
  fi
fi
echo "[install] python=$PY torch-branch=$CUDA"

$PY -m venv .venv
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi
python -m pip install -U pip
if [[ "$CUDA" == "cpu" ]]; then python -m pip install -r requirements/torch-cpu.txt
else python -m pip install -r "requirements/torch-cu${CUDA}.txt"; fi
python -m pip install -r requirements/base.txt
python -m pip install --no-deps -r requirements/extras.txt
python -m pip install -e .

export PYTHONIOENCODING=utf-8
python scripts/00_check_env.py
echo "[install] done. Activate with: source .venv/bin/activate"
