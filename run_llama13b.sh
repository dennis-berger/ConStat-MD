#!/bin/bash
#SBATCH --job-name=codegen_gpu_llama13b  # Job name for Meta-Llama 13B model
#SBATCH --output=output_llama13b.txt     # Output file
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --gres=gpu:rtx4090:1             # Request 1 Nvidia RTX 4090 GPU
#SBATCH --mem=90GB                       # Memory allocation (90GB per RTX 4090 GPU)
#SBATCH --partition=gpu                  # Use the GPU partition
#SBATCH --time=4:00:00                   # Time limit (hh:mm:ss)

# Load CUDA and Python environment
module load CUDA/11.8.0
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
pip install transformers datasets "accelerate>=0.26.0"

# Log in to Hugging Face
echo "Logging into Hugging Face..."
HUGGINGFACE_TOKEN=your_huggingface_access_token
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

# Run the Python script for the specific model
echo "Running code generation script for Meta-Llama 13B with GPU support..."
python /storage/homefs/db18y058/ConStat-MD/main.py --model meta-llama/Llama-2-13b-chat-hf

echo "Job finished successfully."