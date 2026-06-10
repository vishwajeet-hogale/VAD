# VAD on a Slurm cluster

This repository is pinned to an older OpenMMLab stack:

- Python 3.8
- PyTorch 1.9.1 + CUDA 11.1
- mmcv-full 1.4.0
- mmdet 2.14.0
- mmsegmentation 0.14.1
- mmdet3d 0.17.1

The main cluster risk is version drift. Your current cluster only exposes `module load cuda/12.1`, while VAD is pinned to a cu111-era software stack. The helper scripts in this repo now default to `cuda/12.1` to match that environment, but bare-metal setup remains best-effort. The least risky production path is still a container built around the stack above.

## 1. First-time environment setup

Do the first install on a login node or interactive GPU allocation where `conda`, `git`, a C++ compiler, and a CUDA toolkit are available.

```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00
cd /path/to/VAD

export ANACONDA_MODULE=miniconda/24.3.0
export GCC_MODULE=gcc/10.3.0
export CUDA_MODULE=cuda/12.1
export CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh

bash tools/setup_cluster_env.sh
```

Notes:

- The module names above are examples. Replace them with your cluster's actual module names.
- If your cluster does not use environment modules, leave `ANACONDA_MODULE`, `GCC_MODULE`, and `CUDA_MODULE` unset.
- The setup script prints a warning when using `cuda/12.1` because VAD still depends on Torch 1.9.1 + `mmcv-full==1.4.0` wheels built for cu111.
- `requirements.txt` in this repo is a frozen environment snapshot, not the best bootstrap path for a fresh cluster install.

## 2. Put the dataset on shared or scratch storage

The VAD configs expect the dataset under `data/nuscenes/` and the CAN bus expansion under `data/can_bus/`.

```bash
cd /path/to/VAD
mkdir -p data
ln -s /scratch/$USER/datasets/nuscenes data/nuscenes
ln -s /scratch/$USER/datasets/can_bus data/can_bus
```

If the custom annotation pickles are not already present, generate them with:

```bash
conda activate vad
python tools/data_converter/vad_nuscenes_converter.py \
  nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag vad_nuscenes \
  --version v1.0 \
  --canbus ./data
```

## 3. Avoid network downloads during training

The configs use `torchvision://resnet50` by default. On clusters with no outbound internet from compute nodes, pre-download the backbone checkpoint once:

```bash
cd /path/to/VAD
mkdir -p ckpts
wget -O ckpts/resnet50-19c8e357.pth \
  https://download.pytorch.org/models/resnet50-19c8e357.pth
```

Then pass this override when you submit training:

```bash
--cfg-options model.pretrained.img=ckpts/resnet50-19c8e357.pth
```

## 4. Smoke test the install

```bash
cd /path/to/VAD
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate vad

python -c "import torch, mmcv, mmdet, mmdet3d, projects.mmdet3d_plugin; print(torch.cuda.is_available(), torch.__version__, mmcv.__version__, mmdet3d.__version__)"
python -c "from mmcv import Config; cfg = Config.fromfile('projects/configs/VAD/VAD_tiny_stage_1.py'); print(cfg.model.type, cfg.data_root)"
```

If either command fails, fix the environment before submitting a long job.

## 5. Submit a training job

`tools/slurm_train.sh` is a thin wrapper around `tools/train.py --launcher slurm`.

```bash
cd /path/to/VAD
sbatch \
  --export=ALL,ENV_NAME=vad,CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh,CUDA_MODULE=cuda/12.1,PROJECT_ROOT=/path/to/VAD,CONFIG=projects/configs/VAD/VAD_tiny_stage_1.py,WORK_DIR=/scratch/$USER/vad_runs/tiny_stage_1 \
  tools/slurm_train.sh \
  --cfg-options model.pretrained.img=ckpts/resnet50-19c8e357.pth
```

Adjust the `#SBATCH` lines in the script to match your cluster.

Important:

- `--ntasks-per-node` should match the number of GPUs you request on each node.
- For multi-node runs, keep `--launcher slurm`; the wrapper already exports `MASTER_ADDR` and `MASTER_PORT`.
- Start with `VAD_tiny_stage_1.py` or `VAD_base_stage_1.py` before moving to stage 2.

## 6. Submit a single-GPU evaluation job

The VAD docs note that evaluation should stay single-GPU.

```bash
cd /path/to/VAD
sbatch \
  --export=ALL,ENV_NAME=vad,CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh,CUDA_MODULE=cuda/12.1,PROJECT_ROOT=/path/to/VAD,CONFIG=projects/configs/VAD/VAD_tiny_stage_1.py,CHECKPOINT=/scratch/$USER/vad_runs/tiny_stage_1/latest.pth,OUT=/scratch/$USER/vad_eval/vad_tiny_stage_1.pkl \
  tools/slurm_eval.sh
```

## 7. Submit OpenLane-V2 BEV extraction

The inference-first OpenLane-V2 bridge added in this repo uses:

- config: `projects/configs/VAD/VAD_tiny_stage_1_olv2_extract.py`
- extractor: `tools/extract_olv2_bev_embeddings.py`
- Slurm wrapper: `tools/slurm_extract_olv2_bev.sh`

Example train-split extraction:

```bash
cd /path/to/VAD
sbatch \
  --export=ALL,ENV_NAME=vad,CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh,CUDA_MODULE=cuda/12.1,PROJECT_ROOT=/path/to/VAD,CHECKPOINT=/scratch/$USER/vad_ckpts/vad_tiny_stage_1.pth,SPLIT=train,OUTPUT_DIR=/scratch/$USER/GMM_embeddings_vad \
  tools/slurm_extract_olv2_bev.sh
```

Useful overrides:

- `MODES=baseline_meanpool,pos_meanpool`
- `MAX_SAMPLES=64` for a smoke test
- `DISABLE_TEMPORAL=1` to force single-frame extraction

The extractor writes NPZ files compatible with the existing OOD GMM pipeline naming convention, for example `bev_embeddings_train_baseline_meanpool.npz`.

## 8. Common failure modes

- `mmcv-full` install fails: your CUDA and PyTorch pair does not match the wheel index. On this cluster, that usually means the bare-metal `cuda/12.1` path is running into the repo's cu111-era pins.
- `python setup.py develop` for `mmdetection3d` fails: your compiler or CUDA headers are missing, or the host `cuda/12.1` toolkit is incompatible with the pinned Torch 1.9.1 build.
- Training tries to download ResNet-50 on the compute node: add `--cfg-options model.pretrained.img=ckpts/resnet50-19c8e357.pth`.
- The job cannot see the dataset: put the real data on shared or scratch storage and symlink it into `data/`.
- OpenLane-V2 extraction fails with missing annotation files: verify `data_root` points at the directory containing `data_dict_subset_A_train_lanesegnet.pkl` and `data_dict_subset_A_val_lanesegnet.pkl`.