#!/usr/bin/env bash
# Thin launcher: env activation + Python CLI.
# use: bash run_tracer.sh plot
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [ -f "/home/chak803/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/home/chak803/miniforge3/etc/profile.d/conda.sh"
  conda activate /home/chak803/miniforge3/envs/llm_env 2>/dev/null || true
fi

exec python "$SCRIPT_DIR/run_tracer.py" "$@"