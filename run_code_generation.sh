#!/bin/bash
#SBATCH --job-name=codegen    # Job name
#SBATCH --output=output.txt   # Output file
#SBATCH --ntasks=1            # Number of tasks (processes)
#SBATCH --cpus-per-task=4     # Number of CPU cores per task
#SBATCH --mem=8GB             # Memory per CPU
#SBATCH --time=2:00:00        # Time limit (hh:mm:ss)

# Load Python environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate myenv  # Activate your virtual environment

# Run Python script
python /storage/homefs/db18y058/ConStat-MD/main.py
