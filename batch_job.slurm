#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -t 3-00:00          # Runtime in D-HH:MM
#SBATCH --mem=16G

##SBATCH --constraint rhel8  # to avoid the glibc error
#SBATCH -p liulab            
#SBATCH -o download.out          # Write stdout to file named JOBIDNUM.out in current dir
#SBATCH -e download.err          # Write stderr to file named JOBIDNUM.err in current dir

source /cluster/tufts/hpc/apps/rhel8/external/apps/anaconda/202410/etc/profile.d/conda.sh
conda activate geo
python download_hrrr_ne.py 
conda deactivate
