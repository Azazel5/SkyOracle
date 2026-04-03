#!/bin/bash
#
# Subhanga - 7-feature ablation (6 regression + binary BCE), same GPU/hparams as train_subhanga.slurm
#
# Submit from repo root:
#   sbatch subhanga/train_subhanga_7feat.slurm
#

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --constraint="l40|l40s|a100"
#SBATCH -o subhanga/logs/train_7feat_%j.out
#SBATCH -e subhanga/logs/train_7feat_%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CWD:  $(pwd)"

mkdir -p subhanga/logs

cd "${SLURM_SUBMIT_DIR}"

BRANCH="${BRANCH:-CNN-train-subhanga}"
git checkout "${BRANCH}"
git pull

echo "GPU info:"
nvidia-smi || true

export APPTAINER_BINDPATH="/cluster/tufts/c26sp1cs0137/data"

module load pytorch/2.8.0-cuda12.9-cudnn9

python3 --version
python3 -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

DATASET_DIR="${DATASET_DIR:-/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/cluster/tufts/c26sp1cs0137/supadh03/SkyOracle/checkpoints/subhanga_7feat}"

echo "DATASET_DIR=${DATASET_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "[7feat] Same hyperparameters as vanilla; best.pt = lowest val regression MSE only."

python3 -m subhanga.train \
  --dataset-dir "${DATASET_DIR}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --epochs 10 \
  --batch-size 16 \
  --base-channels 16 \
  --multitask7

echo "Job finished: $(date)"
