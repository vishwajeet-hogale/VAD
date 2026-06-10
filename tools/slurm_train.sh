#!/usr/bin/env bash
#SBATCH --job-name=vad-train
#SBATCH --output=logs/vad_train_%j.out
#SBATCH --error=logs/vad_train_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
CONFIG="${CONFIG:-projects/configs/VAD/VAD_tiny_stage_1.py}"
WORK_DIR="${WORK_DIR:-$PROJECT_ROOT/work_dirs/vad_tiny_stage_1}"
ENV_NAME="${ENV_NAME:-vad}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ANACONDA_MODULE="${ANACONDA_MODULE:-}"
GCC_MODULE="${GCC_MODULE:-}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if type module >/dev/null 2>&1; then
    module purge || true
    for module_name in "$ANACONDA_MODULE" "$GCC_MODULE" "$CUDA_MODULE"; do
        if [[ -n "$module_name" ]]; then
            module load "$module_name"
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
mkdir -p logs "$WORK_DIR"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export MASTER_PORT
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

srun python tools/train.py \
    "$CONFIG" \
    --launcher slurm \
    --work-dir "$WORK_DIR" \
    "$@"