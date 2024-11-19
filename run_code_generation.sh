#!/bin/bash
#SBATCH --job-name=codegen    # Job name
#SBATCH --output=output.txt   # Output file
#SBATCH --ntasks=1            # Number of tasks (processes)
#SBATCH --cpus-per-task=4     # Number of CPU cores per task
#SBATCH --mem=16GB            # Memory per CPU
#SBATCH --time=4:00:00        # Time limit

# Load Python and Anaconda
module load Anaconda3
eval "$(conda shell.bash hook)"

# Create a Conda environment (if it doesn't already exist)
ENV_NAME="codegen_env"
if ! conda info --envs | grep -q $ENV_NAME; then
    echo "Creating Conda environment: $ENV_NAME"
    conda create --name $ENV_NAME python=3.8 -y
fi

# Activate the environment
conda activate $ENV_NAME

# Install necessary libraries
echo "Installing required packages..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers datasets

# Run the Python script
echo "Running code generation script..."
python /storage/homefs/db18y058/ConStat-MD/main.py

echo "Job finished successfully."
