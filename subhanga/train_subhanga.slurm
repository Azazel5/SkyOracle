#!/bin/bash
#
# Subhanga - Part 1 CNN training (GPU)
#
# Submit from repo root:
#   sbatch subhanga/train_subhanga.slurm
#
# Optional overrides at submit time:
#   sbatch --export=ALL,DATASET_DIR=/path/to/dataset,CHECKPOINT_DIR=/path/to/ckpts,BRANCH=CNN-train-subhanga subhanga/train_subhanga.slurm
#

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --constraint="l40|l40s|a100"
#SBATCH -o subhanga/logs/train_%j.out
#SBATCH -e subhanga/logs/train_%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CWD:  $(pwd)"

mkdir -p subhanga/logs

# If you submit from the repo root, this will already be correct.
# Otherwise, cd into the directory where you submitted the job.
cd "${SLURM_SUBMIT_DIR}"

# Sync code (safe even if you are already up to date)
BRANCH="${BRANCH:-CNN-train-subhanga}"
git checkout "${BRANCH}"
git pull

echo "GPU info:"
nvidia-smi || true

# --- Environment ---
# NOTE: The cluster PyTorch modules run inside an Apptainer container image.
# The dataset lives under /cluster/tufts/... which is not visible inside that
# container unless we explicitly bind-mount it.
#
# Setting APPTAINER_BINDPATH is safe even if Apptainer isn't used; it is just an
# environment variable. When Apptainer is used, it fixes torch.load(...) file
# lookups for /cluster/tufts/... paths.
export APPTAINER_BINDPATH="/cluster/tufts/c26sp1cs0137/data"

module load pytorch/2.8.0-cuda12.9-cudnn9

python3 --version
python3 -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

# --- Paths ---
DATASET_DIR="${DATASET_DIR:-/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/cluster/tufts/c26sp1cs0137/supadh03/SkyOracle/checkpoints/subhanga}"

echo "DATASET_DIR=${DATASET_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"

# --- Train ---
python3 -m subhanga.train \
  --dataset-dir "${DATASET_DIR}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --epochs 10 \
  --batch-size 16 \
  --base-channels 16

echo "Job finished: $(date)"

