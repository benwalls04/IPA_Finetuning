#!/bin/bash
#SBATCH --job-name=ipa_finetuning_sst2_english
#SBATCH --account=PAS2836
#SBATCH --output=/fs/ess/PAS2836/ipa_gpt/jobs/logs/%x-%j.out
#SBATCH --error=/fs/ess/PAS2836/ipa_gpt/jobs/logs/errors/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus-per-node=1

echo "===== [$(date)] JOB STARTED ====="

# Load required modules
module load miniconda3/24.1.2-py310 cuda/12.4.1

# Properly activate conda and the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ipa_env  # Use your specific environment name
pip install datasets 
pip install transformers
pip install torch
pip install wandb
pip install scikit-learn

# Verify environment
echo "Python: $(which python) ($(python --version))"

# setup paths
scratch_prefix="/fs/scratch/PAS2836/ipa_gpt"
storage_prefix="/fs/ess/PAS2836/ipa_gpt"
datasets_prefix="$storage_prefix/datasets"
checkpoints_prefix="$storage_prefix/checkpoints"
tokenizers_prefix="$storage_prefix/tokenizers"
scratch_datasets_prefix="$scratch_prefix/tokens"
scratch_github_prefix="$scratch_prefix/github"
mkdir -p $scratch_datasets_prefix $scratch_github_prefix $checkpoints_prefix

# repo_name="IPA_Finetuning"
# repo_address="git@github.com:aaron-jencks/$repo_name.git"
# repo_branch="main"
# repo_dir="$scratch_github_prefix/$repo_name"

# if [ ! -d "$repo_dir" ]; then
#   cd "$scratch_github_prefix"
#   git clone "$repo_address"
#   cd "$repo_name"
#   git checkout "$repo_branch"
# else
#   cd "$repo_dir"
#   git fetch origin
#   git checkout "$repo_branch"
#   git reset --hard origin/"$repo_branch"
#   git pull origin "$repo_branch"
# fi

# Parse command line arguments for dataset
if [ "$#" -ne 1 ]; then
    echo "ERROR: Missing dataset argument"
    echo "Usage: sbatch grid_search.sh <dataset_name>"
    exit 1
fi

dataset_name="$1"
echo "Dataset: $dataset_name"

# Script specific names
model="openwebtext_normal_multi_node_12_5"
wandb_project="ipa_finetuning_sst2_english"
parent_dataset="nyu-mll/glue"

checkpoint_path="$checkpoints_prefix/$model/ckpt.pt"
token_data_dir="$scratch_datasets_prefix/$wandb_project"
mkdir -p "$token_data_dir"

# Define hyperparameter grids
batch_sizes=(16)
learning_rates=(3e-5 5e-5 1e-4)
grad_clips=(0.1 0.25 0.5)
warmup_iter_ratios=(0.03 0.06 0.1)

temp_finetune_script="/users/PAS2836/benwalls2004/ipa_finetuning/finetune.py"

# Loop through all combinations
for batch_size in "${batch_sizes[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for grad_clip in "${grad_clips[@]}"; do
      for warmup_iter_ratio in "${warmup_iter_ratios[@]}"; do
        echo "Running with batch_size=${batch_size}, learning_rate=${lr}, grad_clip=${grad_clip}, warmup_iter_ratio=${warmup_iter_ratio}"
        
        # Run Python and capture ALL output (both stdout and stderr)
        output=$(python $temp_finetune_script \
          --dataset "$dataset_name" \
          --parent_dataset "$parent_dataset" \
          --pretrained_ckpt_path "$checkpoint_path" \
          --out_dir "$checkpoints_prefix" \
          --tokenizer_dir "$tokenizers_prefix" \
          --data_dir "$token_data_dir" \
          --hf_cache "$datasets_prefix" \
          --wandb_project "$wandb_project" \
          --wandb_log \
          --num_epochs 2 \
          --hyperparameters batch_size=$batch_size learning_rate=$lr grad_clip=$grad_clip warmup_iter_ratio=$warmup_iter_ratio 2>&1)
        
        # Save the exit code immediately
        exit_code=$?
        
        # Print the output
        echo "$output"
        
        # Check the exit code
        if [ $exit_code -ne 0 ]; then
          echo "ERROR: Run failed with batch_size=$batch_size, learning_rate=$lr, grad_clip=$grad_clip, warmup_iter_ratio=$warmup_iter_ratio"
        else
          echo "===== [$(date)] Completed run with batch_size=$batch_size, learning_rate=$lr, grad_clip=$grad_clip, warmup_iter_ratio=${warmup_iter_ratio} ====="
        fi
      done
    done
  done
done

echo "===== [$(date)] JOB COMPLETED ====="
