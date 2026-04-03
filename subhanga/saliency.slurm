#!/bin/bash
#
# Saliency map generation for Part 2
#
# Submit from repo root:
#   sbatch subhanga/saliency.slurm

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --constraint="l40|l40s|a100"
#SBATCH -o subhanga/logs/saliency_%j.out
#SBATCH -e subhanga/logs/saliency_%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CWD:  $(pwd)"

mkdir -p subhanga/logs subhanga/figures

export APPTAINER_BINDPATH="/cluster/tufts/c26sp1cs0137/data"

module load pytorch/2.8.0-cuda12.9-cudnn9

# PyTorch module does not include cartopy; install into user site for this interpreter
echo "Installing cartopy (user) if needed …"
python3 -m pip install --user --quiet cartopy

python3 --version
python3 -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

python3 subhanga/saliency.py

echo "Job finished: $(date)"
