#!/usr/bin/env bash
#SBATCH --job-name=vad-eval
#SBATCH --output=logs/vad_eval_%j.out
#SBATCH --error=logs/vad_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
CONFIG="${CONFIG:-projects/configs/VAD/VAD_tiny_stage_1.py}"
CHECKPOINT="${CHECKPOINT:-}"
OUT="${OUT:-$PROJECT_ROOT/results/vad_eval.pkl}"
ENV_NAME="${ENV_NAME:-vad}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ANACONDA_MODULE="${ANACONDA_MODULE:-}"
GCC_MODULE="${GCC_MODULE:-}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"

if [[ -z "$CHECKPOINT" ]]; then
    echo "Set CHECKPOINT to the model file you want to evaluate."
    exit 1
fi

if type module >/dev/null 2>&1; then
    module purge || true
    for module_name in "$ANACONDA_MODULE" "$GCC_MODULE" "$CUDA_MODULE"; do
        if [[ -n "$module_name" ]]; then
            if ! module load "$module_name"; then
                if [[ "$module_name" == "$ANACONDA_MODULE" ]] && [[ -f "$CONDA_SH" ]]; then
                    echo "WARNING: module '$module_name' was not found; using CONDA_SH=$CONDA_SH instead." >&2
                else
                    echo "ERROR: Unable to load required module '$module_name'." >&2
                    exit 1
                fi
            fi
        fi
    done
fi

if [[ ! -f "$CONDA_SH" ]]; then
    echo "conda.sh not found at $CONDA_SH"
    exit 1
fi

source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$PROJECT_ROOT"
mkdir -p logs "$(dirname "$OUT")"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

python tools/test.py \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher none \
    --eval bbox \
    --out "$OUT" \
    "$@"