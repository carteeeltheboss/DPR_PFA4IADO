#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
VENV_DIR="$ROOT_DIR/DPR_MedFusionNet/venv"
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
APP_FILE="$ROOT_DIR/DPR_WebService/app.py"

if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
  echo "Virtual environment not found. Run ./run_first.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$ACTIVATE_SCRIPT"

cd "$ROOT_DIR"
exec python "$APP_FILE"
