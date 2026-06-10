#!/bin/bash
#SBATCH --job-name=vad-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=logs/vad_eval_%j.out
#SBATCH --error=logs/vad_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

# Load CUDA module
module load cuda/12.1

# Activate conda environment
set +u
source ~/.bashrc
set -u
conda activate lanesegnet

PROJECT_ROOT="${PROJECT_ROOT:-/home/hogale.v/Lane_Topology/VAD}"
CONFIG="${CONFIG:-projects/configs/VAD/VAD_tiny_stage_1.py}"
CHECKPOINT="${CHECKPOINT:-}"
OUT="${OUT:-$PROJECT_ROOT/results/vad_eval.pkl}"

if [[ -z "$CHECKPOINT" ]]; then
    echo "Set CHECKPOINT to the model file you want to evaluate."
    exit 1
fi

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