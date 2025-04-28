#!/bin/bash
#SBATCH --job-name=ipa_grid_search_sst2_english
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
conda init bash
conda activate base

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

repo_name="IPA_Finetuning"
repo_address="git@github.com:benwalls04/$repo_name.git"
repo_branch="Common_Changes"
repo_dir="$scratch_github_prefix/$repo_name"

if [ ! -d "$repo_dir" ]; then
  cd "$scratch_github_prefix"
  git clone "$repo_address"
  cd "$repo_name"
  git checkout "$repo_branch"
else
  cd "$repo_dir"
  git pull
fi

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
wandb_project="ipa_grid_search_${dataset_name}_english"
parent_dataset="nyu-mll/glue"

checkpoint_path="$checkpoints_prefix/$model/ckpt.pt"
token_data_dir="$scratch_datasets_prefix/$wandb_project"
mkdir -p "$token_data_dir"

# Define hyperparameter grids
batch_sizes=(16 32 64)
learning_rates=(1e-5 2e-5 3e-5 5e-5)
weight_decays=(0.0 0.01 0.1)

# Loop through all combinations
for batch_size in "${batch_sizes[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for wd in "${weight_decays[@]}"; do
      echo "Running with batch_size=${batch_size}, learning_rate=${lr}, weight_decay=${wd}"
      
      python finetune.py \
        --dataset "$dataset_name" \
        --parent_dataset "$parent_dataset" \
        --pretrained_ckpt_path "$checkpoint_path" \
        --out_dir "$checkpoints_prefix" \
        --tokenizer_dir "$tokenizers_prefix" \
        --data_dir "$token_data_dir" \
        --hf_cache "$datasets_prefix" \
        --wandb_project "$wandb_project" \
        --wandb_log \
        --dont_save_ckpt \
        --num_epochs 2 \
        --hyperparameters "batch_size=$batch_size" "learning_rate=$lr" "weight_decay=$wd"
      
      if [ $? -ne 0 ]; then
        echo "ERROR: Run failed with batch_size=$batch_size, learning_rate=$lr, weight_decay=${wd}"
      else
        echo "===== [$(date)] Completed run with batch_size=$batch_size, learning_rate=$lr, weight_decay=${wd} ====="
      fi
    done
  done
done

echo "===== [$(date)] JOB COMPLETED ====="
