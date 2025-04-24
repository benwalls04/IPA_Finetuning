import argparse
from datasets import load_dataset
from tasks.classification_sst2 import ClassificationSST2
import os
import wandb
import json
import torch
import time
from model import GPTConfig, GPT
import numpy as np
from contextlib import nullcontext
import math
import torch.nn as nn
import torch.nn.functional as F
import inspect

datasets = {
  "sst2": ClassificationSST2
}

# static parameters (DONT CHANGE)
absolute_paths = True
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
device = 'cuda' 
dtype = 'float16' if torch.cuda.is_available() else 'float32' 
backend = 'nccl' 
storage_prefix = "./"
if absolute_paths:
    storage_prefix = "/users/PAS2836/benwalls2004/ipa_finetuning"

out_dir = f"{storage_prefix}/checkpoints"
tokenizer_dir = f"{storage_prefix}/tokenizers"
data_dir = f"{storage_prefix}/data"

gradient_accumulation_steps = 4
num_epochs = 10
dropout = 0.0
wandb_project = 'ipa_finetuning'
log_interval = 5
eval_interval = 5
eval_iters = 40

always_save_checkpoint = False
bias = False
wandb_log = True
decay_lr = True
force_cuda = True
eval_only = False 

# Update tokenizer path in ClassificationSST2
vocab_file = f"{tokenizer_dir}/bpe-normal-number-preservation-vocab.json"
merges_file = f"{tokenizer_dir}/bpe-normal-number-preservation-merges.txt"

def configure_optimizers(device_type, model): 
    # First, separate parameters into pretrained model and classification head
    pretrained_params = list(model.pretrained_model.parameters())
    pretrained_param_ids = set(id(p) for p in pretrained_params)
    
    # Get all parameters that are not part of the pretrained model
    classifier_params = [p for p in model.parameters() if id(p) not in pretrained_param_ids]
    
    # Create parameter dictionaries for each group
    pretrained_param_dict = {f"pretrained.{i}": p for i, p in enumerate(pretrained_params) if p.requires_grad}
    classifier_param_dict = {f"classifier.{i}": p for i, p in enumerate(classifier_params) if p.requires_grad}
    
    # Apply the 2D weight decay rule to each group separately
    pretrained_decay_params = [p for n, p in pretrained_param_dict.items() if p.dim() >= 2]
    pretrained_nodecay_params = [p for n, p in pretrained_param_dict.items() if p.dim() < 2]
    
    classifier_decay_params = [p for n, p in classifier_param_dict.items() if p.dim() >= 2]
    classifier_nodecay_params = [p for n, p in classifier_param_dict.items() if p.dim() < 2]
    
    # Create optimizer groups with different learning rates
    optim_groups = [
        {'params': pretrained_decay_params, 'weight_decay': model.weight_decay, 'lr': model.learning_rate * 0.1},  # Lower LR for pretrained
        {'params': pretrained_nodecay_params, 'weight_decay': 0.0, 'lr': model.learning_rate * 0.1},  # Lower LR for pretrained
        {'params': classifier_decay_params, 'weight_decay': model.weight_decay},  # Default LR for classifier
        {'params': classifier_nodecay_params, 'weight_decay': 0.0}  # Default LR for classifier
    ]
    
    # Log parameter counts
    print(f"Pretrained model decay params: {len(pretrained_decay_params)}, with {sum(p.numel() for p in pretrained_decay_params):,} parameters")
    print(f"Pretrained model no-decay params: {len(pretrained_nodecay_params)}, with {sum(p.numel() for p in pretrained_nodecay_params):,} parameters")
    print(f"Classifier decay params: {len(classifier_decay_params)}, with {sum(p.numel() for p in classifier_decay_params):,} parameters")
    print(f"Classifier no-decay params: {len(classifier_nodecay_params)}, with {sum(p.numel() for p in classifier_nodecay_params):,} parameters")
    
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=model.learning_rate, betas=(model.beta1, model.beta2), **extra_args)
    print(f"using fused AdamW: {use_fused}")
    
    return optimizer

def get_lr(model, it):
    # 1) linear warmup for warmup_iters steps
    if it < model.warmup_iters:
        return model.learning_rate * (it + 1) / (model.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > model.lr_decay_iters:
        return model.min_lr
    # 3) in between, use linear decay down to min learning rate
    decay_ratio = (it - model.warmup_iters) / (model.lr_decay_iters - model.warmup_iters)
    assert 0 <= decay_ratio <= 1
    # Linear decay (instead of cosine)
    return model.learning_rate - decay_ratio * (model.learning_rate - model.min_lr)

# get the dataset from the command line
parser = argparse.ArgumentParser(description='Finetune a model on a dataset')
parser.add_argument('--dataset', type=str, required=True,
                  help='Path to the dataset or dataset name')
args = parser.parse_args()
dataset_arg = args.dataset
if dataset_arg not in datasets:
    raise ValueError(f"Dataset {dataset_arg} not found. Please choose from: {datasets}")

# load the datatset and needed functions 
dataset = load_dataset("glue", dataset_arg)
model_class = datasets[dataset_arg]
model = model_class(device)

train_file_path = f"{data_dir}/{dataset_arg}/train.bin"
val_file_path = f"{data_dir}/{dataset_arg}/val.bin"
metadata_file_path = f"{data_dir}/{dataset_arg}/metadata.json"

# preprocess the dataset
if not os.path.exists(train_file_path):
    os.makedirs(f"{data_dir}/{dataset_arg}", exist_ok=True)

    # tokenize the datatset 
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    train_dataset = train_dataset.map(model.preprocess, batched=True)
    validation_dataset = validation_dataset.map(model.preprocess, batched=True)

    # prepare the data 
    train_data, val_data, metadata = model.prepare_data(train_dataset, validation_dataset)

    # save the data 
    train_data.tofile(train_file_path)
    val_data.tofile(val_file_path)
    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f)

# setup the GPUs
if force_cuda and not torch.cuda.is_available():
    raise Exception('CUDA not available')
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

meta_vocab_size = None
with open(metadata_file_path, 'r') as f:
    meta = json.load(f)
meta_vocab_size = meta['vocab_size']
meta_block_size = meta['max_length']

# load the pretrained model
print(f"Resuming training from {out_dir}")
ckpt_path = os.path.join(out_dir, 'openwebtext_normal_multi_node_12_5/ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=meta_block_size,
                bias=bias, vocab_size=meta_vocab_size, dropout=dropout) 
for k in ['n_layer', 'n_head', 'n_embd', 'bias', 'block_size', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
  
gptconf = GPTConfig(**model_args)
pretrained_model = GPT(gptconf)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model_dict = pretrained_model.state_dict()
# only load the parameters that match the checkpoint weights 
filtered_state_dict = {k: v for k, v in state_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(filtered_state_dict)
pretrained_model.load_state_dict(model_dict)
pretrained_model.to(device)
model.pretrained_model = pretrained_model
model.to(device)  # Move the entire model to the device

# set up the optimizer (from pretrained if possible)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Fix the parameter separation logic
pretrained_params = set(model.pretrained_model.parameters())
new_head_params = []

for param in model.parameters():
    # Check if this parameter object is the same object as any in pretrained_params
    if not any(param is p for p in pretrained_params):
        new_head_params.append(param)

optimizer = configure_optimizers(device_type, model)

if wandb_log:
    config = {
        "learning_rate": model.learning_rate,
        "weight_decay": model.weight_decay,
        "beta1": model.beta1,
        "beta2": model.beta2,
        "grad_clip": model.grad_clip,
        "dropout": model.dropout,
        "warmup_iter_ratio": model.warmup_iter_ratio,
        "lr_decay_iter_ratio": model.lr_decay_iter_ratio,
        "min_lr": model.min_lr,
        "num_epochs": num_epochs,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    wandb.init(project=wandb_project, name=f"{dataset_arg}-{time.time()}", config=config)

# calculate number of iterations required
tokens_per_iter = gradient_accumulation_steps * model.batch_size * block_size
data = np.memmap(train_file_path, dtype=np.uint16, mode='r')
token_count = len(data)
max_iters = int(np.ceil(token_count / tokens_per_iter)) * num_epochs
model.warmup_iters = int(model.warmup_iter_ratio * max_iters)
model.lr_decay_iters = int(model.lr_decay_iter_ratio * max_iters)

best_val_loss = float('inf')
best_checkpoint_path = f"{out_dir}/{dataset_arg}-ckpt.pt"

# Check if a previous best checkpoint exists
if os.path.exists(best_checkpoint_path):
    print(f"Loading previous best checkpoint from {best_checkpoint_path}")
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    best_val_loss = best_checkpoint.get('val_loss', float('inf'))
    print(f"Previous best validation loss: {best_val_loss:.4f}")

# training loop
get_batch = model.get_batch
X, Y = get_batch('train') 
iter_num = 0

for param in model.pretrained_model.parameters():
    param.requires_grad = True 

while True:
    lr = get_lr(model, iter_num) if decay_lr else model.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints    
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        
        X, Y = get_batch('train')
        
        scaler.scale(loss).backward()

    if model.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # evaluate the loss on train/val sets and write checkpoints    
    if iter_num % eval_interval == 0:
        losses = model.estimate_loss(ctx, eval_iters)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val f2 {losses['val_f2']:.4f}, val accuracy {losses['val_accuracy']:.4f}")
        
        # Check if this is the best validation loss so far
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            print(f"New best validation loss: {best_val_loss:.4f}, saving checkpoint to {best_checkpoint_path}")
            
            # Save the best model checkpoint
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'val_loss': best_val_loss,
                'val_accuracy': losses['val_accuracy'],
                'val_f2': losses['val_f2']
            }
            torch.save(checkpoint, best_checkpoint_path)
        
        if wandb_log:            
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "train/f2": losses['train_f2'],
                "val/f2": losses['val_f2'],
                "lr": lr,
                "train/accuracy": losses['train_accuracy'],
                "val/accuracy": losses['val_accuracy'],
                "best_val_loss": best_val_loss
            })

    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break