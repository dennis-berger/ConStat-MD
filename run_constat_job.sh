#!/bin/bash
#SBATCH --job-name=constat_analysis       # Job name
#SBATCH --output=constat_output.txt       # Output file for logs
#SBATCH --error=constat_error.txt         # Error file for logs
#SBATCH --time=02:00:00                   # Maximum runtime (hh:mm:ss)
#SBATCH --mem=16G                         # Memory allocation
#SBATCH --cpus-per-task=4                 # Number of CPUs
#SBATCH --ntasks=1                        # Single task

# Load Conda module (adjust based on your cluster's configuration)
module load Anaconda3  # Load Conda module (if required)

# Set up Conda environment for ConStat
if ! conda info --envs | grep -q constat; then
    conda create -n constat python=3.10 -y
fi
conda activate constat

python -m pip install -e .

# Run the Python analysis script
python run_constat.py
