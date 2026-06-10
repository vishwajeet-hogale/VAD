#!/usr/bin/env bash
#SBATCH --job-name=vad-olv2-bev
#SBATCH --output=logs/vad_olv2_bev_%j.out
#SBATCH --error=logs/vad_olv2_bev_%j.err
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
CONFIG="${CONFIG:-projects/configs/VAD/VAD_tiny_stage_1_olv2_extract.py}"
CHECKPOINT="${CHECKPOINT:-}"
SPLIT="${SPLIT:-train}"
MODES="${MODES:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/GMM_embeddings_vad}"
ENV_NAME="${ENV_NAME:-lanesegnet}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DISABLE_TEMPORAL="${DISABLE_TEMPORAL:-0}"

if [[ -z "$CHECKPOINT" ]]; then
    echo "Set CHECKPOINT to the VAD model checkpoint you want to extract from."
    exit 1
fi

module load "$CUDA_MODULE"
set +u
source ~/.bashrc
set -u

if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found after sourcing ~/.bashrc." >&2
    exit 1
fi

conda activate "$ENV_NAME"

cd "$PROJECT_ROOT"
mkdir -p logs "$OUTPUT_DIR"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

CMD=(
    python tools/extract_olv2_bev_embeddings.py
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --split "$SPLIT"
    --modes "$MODES"
    --output-dir "$OUTPUT_DIR"
)

if [[ -n "$MAX_SAMPLES" ]]; then
    CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$DISABLE_TEMPORAL" == "1" ]]; then
    CMD+=(--disable-temporal)
fi

"${CMD[@]}" "$@"