#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
VENV_DIR="$ROOT_DIR/DPR_MedFusionNet/venv"
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
MED_REQ="$ROOT_DIR/DPR_MedFusionNet/requirements.txt"
WEB_REQ="$ROOT_DIR/DPR_WebService/requirements.txt"
APP_FILE="$ROOT_DIR/DPR_WebService/app.py"

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
    return
  fi

  if command -v python >/dev/null 2>&1; then
    printf '%s\n' "python"
    return
  fi

  echo "Python 3 is required but was not found on PATH." >&2
  exit 1
}

if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
  "$(find_python)" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$ACTIVATE_SCRIPT"

python -m pip install --upgrade pip
python -m pip install -r "$MED_REQ"
python -m pip install -r "$WEB_REQ"

cd "$ROOT_DIR"
exec python "$APP_FILE"
