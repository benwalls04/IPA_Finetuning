#!/bin/bash
#SBATCH --job-name=gpt2_finetune
#SBATCH --account=PAS2836
#SBATCH --output=/fs/ess/PAS2836/ipa_gpt/jobs/logs/%x-%j.out
#SBATCH --error=/fs/ess/PAS2836/ipa_gpt/jobs/logs/errors/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus-per-node=1

# Load required modules
module load miniconda3 cuda/12.4.1
conda init bash
conda_env="nanogpt_cu124"
conda activate "$conda_env"
echo "conda environment: $conda_env"

# Display memory info
free --giga
export PYTHONPATH=.

# Setup paths
storage_prefix="/fs/ess/PAS2836/ipa_gpt"
scratch_prefix="/fs/scratch/PAS2836/ipa_gpt"
scratch_github_prefix="$scratch_prefix/github"

# Create necessary directories
mkdir -pv "$scratch_github_prefix"

# Setup github repo
repo_branch_name="hf-rework"
repo_name="ipa-gpt"
repo="git@github.com:aaron-jencks/${repo_name}.git"
repo_prefix="$scratch_github_prefix/$repo_name"

echo "Setting up github repo..."
if [ ! -d "$repo_prefix" ]; then
    echo "Repo did not exist"
    echo "Fetching github repo..."
    cd $scratch_github_prefix
    git clone $repo
    cd $repo_prefix
else
    cd $repo_prefix
    echo "Repo existed"
    echo "Attempting pull..."
    git pull
fi

echo "Checking out specific branch $repo_branch_name"
git checkout "$repo_branch_name"
echo "Repo setup at $repo_prefix"

# Parse command line arguments for dataset
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

dataset_name="$1"

# Setup checkpoint directory
checkpoint_dir="/fs/ess/PAS2836/checkpoints"
mkdir -p "$checkpoint_dir"

# If you need to copy checkpoints from somewhere else, do it here
# cp /path/to/source/checkpoint.pt $checkpoint_dir/

# Run the finetuning script
echo "Starting finetuning on dataset: $dataset_name"
python finetune.py --dataset "$dataset_name" 