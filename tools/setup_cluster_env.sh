#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="${ENV_NAME:-vad}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ANACONDA_MODULE="${ANACONDA_MODULE:-}"
GCC_MODULE="${GCC_MODULE:-}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"
MMDET3D_DIR="${MMDET3D_DIR:-$PROJECT_ROOT/../mmdetection3d}"
TORCH_WHL_INDEX="${TORCH_WHL_INDEX:-https://download.pytorch.org/whl/torch_stable.html}"
MMCV_WHL_INDEX="${MMCV_WHL_INDEX:-https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html}"

if [[ "$CUDA_MODULE" == "cuda/12.1" ]]; then
    echo "WARNING: bare-metal setup under cuda/12.1 is best-effort for this cu111-era VAD stack." >&2
    echo "WARNING: prefer a containerized runtime if your cluster supports it." >&2
fi

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
    echo "Set CONDA_SH to your conda profile script and rerun."
    exit 1
fi

source "$CONDA_SH"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
    torch==1.9.1+cu111 \
    torchvision==0.10.1+cu111 \
    torchaudio==0.9.1 \
    -f "$TORCH_WHL_INDEX"

python -m pip install mmcv-full==1.4.0 -f "$MMCV_WHL_INDEX"

python -m pip install \
    mmdet==2.14.0 \
    mmsegmentation==0.14.1 \
    timm==0.6.12 \
    nuscenes-devkit==1.1.9 \
    terminaltables==3.1.10 \
    descartes==1.1.0 \
    prettytable==3.7.0

if [[ ! -d "$MMDET3D_DIR/.git" ]]; then
    git clone --depth 1 --branch v0.17.1 https://github.com/open-mmlab/mmdetection3d.git "$MMDET3D_DIR"
fi

(
    cd "$MMDET3D_DIR"
    git checkout -f v0.17.1
    python setup.py develop
)

mkdir -p "$PROJECT_ROOT/ckpts" "$PROJECT_ROOT/data"

cat <<EOF
Environment is ready.

Next steps:
  1. Put nuScenes under $PROJECT_ROOT/data/nuscenes or symlink it there.
  2. Put can_bus under $PROJECT_ROOT/data/can_bus.
  3. Optionally predownload ResNet-50:
     wget -O $PROJECT_ROOT/ckpts/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
EOF