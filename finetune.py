import argparse
import os
import pathlib
import time

from contextlib import nullcontext
from datasets import load_dataset, load_from_disk
import inspect
import numpy as np
import torch
import wandb

from tasks.classification_sst2 import ClassificationSST2
from model import GPTConfig, GPT

datasets = {
    "sst2": ClassificationSST2
}

parser = argparse.ArgumentParser(description="Finetune a model on a classification task")

# Required
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--pretrained_ckpt_path', type=pathlib.Path, required=True)

# Dataset settings
parser.add_argument('--parent_dataset', type=str, default='nyu-mll/glue')
parser.add_argument('--no_subset', action='store_true')
parser.add_argument('--from_disk', action='store_true')
parser.add_argument('--use_ipa', action='store_true')
parser.add_argument('--force_tokenization', action='store_true')

# Paths
parser.add_argument('--hf_cache', type=pathlib.Path, default=pathlib.Path('./cache'))
parser.add_argument('--out_dir', type=pathlib.Path, default=pathlib.Path('./checkpoints'))
parser.add_argument('--tokenizer_dir', type=pathlib.Path, default=pathlib.Path('./tokenizers'))
parser.add_argument('--data_dir', type=pathlib.Path, default=pathlib.Path('./datasets'))
parser.add_argument('--tokenizer_name', type=str, default='bpe-normal-number-preservation')

# Training settings
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--eval_iters', type=int, default=40)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)

# Model architecture
parser.add_argument('--n_layer', type=int, default=12)
parser.add_argument('--n_head', type=int, default=12)
parser.add_argument('--n_embd', type=int, default=768)
parser.add_argument('--block_size', type=int, default=1024)

# Device settings
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'])
parser.add_argument('--backend', type=str, default='nccl')

# Logging
parser.add_argument('--wandb_project', type=str, default='ipa_finetuning')
parser.add_argument('--log_interval', type=int, default=5)
parser.add_argument('--eval_interval', type=int, default=5)
parser.add_argument('--wandb_log', action='store_true')
parser.add_argument('--eval_only', action='store_true')

args = parser.parse_args()

always_save_checkpoint = False
bias = False
decay_lr = True
force_cuda = True

# Update tokenizer path in ClassificationSST2
vocab_file = args.tokenizer_dir / f"{args.tokenizer_name}-vocab.json"
merges_file = args.tokenizer_dir / f"{args.tokenizer_name}-merges.txt"

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

dataset_arg = args.dataset
if dataset_arg not in datasets:
    raise ValueError(f"Dataset {dataset_arg} not found. Please choose from: {datasets}")

# load the datatset and needed functions
if args.no_subset:
    if args.from_disk:
        dataset = load_from_disk(args.parent_dataset)
    else:
        dataset = load_dataset(args.parent_dataset)
else:
    if args.from_disk:
        raise Exception('can\'t load from disk with subset.')
    dataset = load_dataset(args.parent_dataset, dataset_arg, cache_dir=str(args.hf_cache))
model_class = datasets[dataset_arg]
model = model_class(args.device, vocab_file, merges_file, args.data_dir, ipa=args.use_ipa)

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
model.prepare_if_needed(train_dataset, validation_dataset, args.force_tokenization)

# setup the GPUs
if force_cuda and not torch.cuda.is_available():
    raise Exception('CUDA not available')
os.makedirs(args.out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in args.device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

meta = model.get_metadata()
meta_vocab_size = meta['vocab_size']
meta_block_size = meta['max_length']

# load the pretrained model
print(f"Resuming training from {args.out_dir}")
checkpoint = torch.load(args.pretrained_ckpt_path, map_location=args.device)
checkpoint_model_args = checkpoint['model_args']

model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=meta_block_size,
                bias=bias, vocab_size=meta_vocab_size, dropout=args.dropout)
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
pretrained_model.to(args.device)
model.pretrained_model = pretrained_model
model.to(args.device)  # Move the entire model to the device

# set up the optimizer (from pretrained if possible)
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

# Fix the parameter separation logic
pretrained_params = set(model.pretrained_model.parameters())
new_head_params = []

for param in model.parameters():
    # Check if this parameter object is the same object as any in pretrained_params
    if not any(param is p for p in pretrained_params):
        new_head_params.append(param)

optimizer = configure_optimizers(device_type, model)

if args.wandb_log:
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
        "num_epochs": args.num_epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    wandb.init(project=args.wandb_project, name=f"{dataset_arg}-{time.time()}", config=config)

# calculate number of iterations required
tokens_per_iter = args.gradient_accumulation_steps * model.batch_size * args.block_size
token_count = model.get_token_count()
max_iters = int(np.ceil(token_count / tokens_per_iter)) * args.num_epochs
model.warmup_iters = int(model.warmup_iter_ratio * max_iters)
model.lr_decay_iters = int(model.lr_decay_iter_ratio * max_iters)

best_val_loss = float('inf')
best_checkpoint_path = f"{args.out_dir}/{dataset_arg}-ckpt.pt"

# Check if a previous best checkpoint exists
if os.path.exists(best_checkpoint_path):
    print(f"Loading previous best checkpoint from {best_checkpoint_path}")
    best_checkpoint = torch.load(best_checkpoint_path, map_location=args.device)
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
    if iter_num == 0 and args.eval_only:
        break

    for micro_step in range(args.gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / args.gradient_accumulation_steps
        
        X, Y = get_batch('train')
        
        scaler.scale(loss).backward()

    if model.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # evaluate the loss on train/val sets and write checkpoints    
    if iter_num % args.eval_interval == 0:
        losses = model.estimate_loss(ctx, args.eval_iters)
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
        
        if args.wandb_log:
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