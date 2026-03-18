#!/bin/bash
#SBATCH -n 1                    # Number of CPU cores
#SBATCH -t 3-00:00              # Runtime: 3 days (D-HH:MM)
#SBATCH --mem=64G               # Memory (need ~800 MB per 24-step chunk)
#SBATCH -p liulab               # Partition
#SBATCH -o log_dataset.out      # stdout log
#SBATCH -e log_dataset.err      # stderr log

source /cluster/tufts/hpc/apps/rhel8/external/apps/anaconda/202410/etc/profile.d/conda.sh
conda activate earth2mip

cd /cluster/tufts/c26sp1cs0137/data/assignment2_data
python generate_dataset.py

conda deactivate
