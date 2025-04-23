#!/bin/bash
#SBATCH --job-name=ipa_finetune
#SBATCH --account=PAS2836
#SBATCH --output=/users/PAS2836/benwalls2004/ipa_finetuning/jobs/logs/%x-%j.out
#SBATCH --error=/users/PAS2836/benwalls2004/ipa_finetuning/jobs/logs/errors/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus-per-node=1

echo "===== [$(date)] JOB STARTED ====="

# Load required modules
module load cuda/11.8.0

# Use your personal conda installation
export PATH="/users/PAS2836/benwalls2004/miniconda3/bin:$PATH"
source "/users/PAS2836/benwalls2004/miniconda3/etc/profile.d/conda.sh"

# Activate the base environment
conda activate base
echo "Python: $(which python) ($(python --version))"

# Setup paths
storage_prefix="/users/PAS2836/benwalls2004/ipa_finetuning"

# Set PYTHONPATH to include the repo directory
export PYTHONPATH="$storage_prefix:$PYTHONPATH"

# Parse command line arguments for dataset
if [ "$#" -ne 1 ]; then
    echo "ERROR: Missing dataset argument"
    echo "Usage: sbatch finetune.sh <dataset_name>"
    exit 1
fi

dataset_name="$1"
echo "Dataset: $dataset_name"

# Setup checkpoint directory
checkpoint_dir="$storage_prefix/checkpoints"
mkdir -p "$checkpoint_dir"

# Change to the correct directory before running the script
cd "$storage_prefix"

# Check if finetune.py exists
if [ ! -f "$storage_prefix/finetune.py" ]; then
    echo "ERROR: finetune.py not found at $storage_prefix/finetune.py"
    exit 1
fi

echo "===== [$(date)] RUNNING PYTHON SCRIPT ====="

# Run the actual script
python finetune.py --dataset "$dataset_name"

echo "===== [$(date)] JOB COMPLETED ====="