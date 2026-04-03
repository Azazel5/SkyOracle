#!/bin/bash
#
# Saliency map generation for Part 2 (CPU — use when gpu queue is backed up)
#
# Same outputs as subhanga/saliency.slurm:
#   logs:    subhanga/logs/saliency_cpu_%j.{out,err}
#   figures: subhanga/figures/saliency_2021.png, mean_sal_2021.npy  (from saliency.py)
#
# Submit from repo root:
#   sbatch subhanga/saliency_cpu.slurm

#SBATCH -p batch
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -J saliency-cpu
#SBATCH -o subhanga/logs/saliency_cpu_%j.out
#SBATCH -e subhanga/logs/saliency_cpu_%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CWD:  $(pwd)"

mkdir -p subhanga/logs subhanga/figures

export APPTAINER_BINDPATH="/cluster/tufts/c26sp1cs0137/data"
export CUDA_VISIBLE_DEVICES=""

module load pytorch/2.8.0-cuda12.9-cudnn9

python3 --version
python3 -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

python3 subhanga/saliency.py

echo "Job finished: $(date)"
