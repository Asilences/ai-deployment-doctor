#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this script inside Linux/WSL2.' >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
# Keep the environment on the Linux filesystem for reliable permissions and I/O.
lab_venv="${INFERENCE_LAB_VENV:-$HOME/.venvs/inference-lab}"
python3 -m venv "$lab_venv"
"$lab_venv/bin/python" -m pip install -r requirements-linux.txt
"$lab_venv/bin/python" scripts/download_model.py
mkdir -p setup_records
"$lab_venv/bin/python" -m pip freeze > setup_records/pip-freeze.txt
"$lab_venv/bin/python" -m inference_lab doctor
echo "Environment ready. Activate with: source $lab_venv/bin/activate"
echo 'Next: python -m inference_lab baseline'
