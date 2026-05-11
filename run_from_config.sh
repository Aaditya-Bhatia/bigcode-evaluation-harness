#!/usr/bin/env bash
# Run HumanEvalFix generation against an already-running vLLM OpenAI-compatible server.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-}"
if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
    echo "Usage: $(basename "$0") <config.yaml>" >&2
    exit 1
fi

_conda_bin="${CONDA_EXE:-$(command -v conda 2>/dev/null || echo "$HOME/miniconda3/bin/conda")}"
if [[ -x "$_conda_bin" ]]; then
    eval "$("$_conda_bin" shell.bash hook)"
else
    echo "Error: conda not found. Set CONDA_EXE or add conda to PATH." >&2
    exit 1
fi
if ! conda activate humanevalfix 2>/dev/null; then
    if ! conda activate SFT_env 2>/dev/null; then
        conda activate vllm_env
    fi
fi

cd "$SCRIPT_DIR"
exec python scripts/run_vllm_humanevalfix_from_config.py "$CONFIG_FILE"
