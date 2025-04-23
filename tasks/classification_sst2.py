from transformers import GPT2TokenizerFast
import json
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from sklearn.metrics import fbeta_score, accuracy_score
import inspect

n_embd = 768
block_size = 1024

class ClassificationSST2(nn.Module):
  def __init__(self, device):
    super().__init__()
    self.pretrained_model = None
    self.device = device
    self.n_layer = 12 

    self.tokenizer = GPT2TokenizerFast(
        vocab_file="./tokenizers/bpe-normal-number-preservation-vocab.json",
        merges_file="./tokenizers/bpe-normal-number-preservation-merges.txt",
        pad_token="<pad>",
        model_max_length=1024,
        padding_side="right",
        truncation=True,
        truncation_side="right"
    )

    self.num_classes = 2
    self.learning_rate = 5e-4
    self.weight_decay = 0.01
    self.beta1 = 0.9
    self.beta2 = 0.999
    self.grad_clip = .1
    self.min_lr = 6e-5
    self.dropout = 0.1
    self.batch_size = 32
    self.warmup_iter_ratio = .05
    self.lr_decay_iter_ratio = .90
    self.warmup_iters = None
    self.lr_decay_iters = None

    self.layer_norm = nn.LayerNorm(n_embd)

    self.classifier = nn.Sequential(
      nn.Linear(n_embd, 256),
      nn.ReLU(),
      nn.Dropout(self.dropout),
      nn.Linear(256, self.num_classes)
    )
        
    with torch.no_grad():
        self.classifier[0].weight.data.normal_(mean=0.0, std=0.02)
        self.classifier[0].bias.data.zero_()

    for pn, p in self.classifier.named_parameters():
      if pn.endswith('c_proj.weight'):
          torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layer))

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def preprocess(self, examples):
    result = self.tokenizer(
        examples["sentence"], 
        padding="max_length", 
        truncation=True, 
        max_length=1024
    )
    result["labels"] = examples["label"]
    return result

  def prepare_data(self, train_dataset, val_dataset):
    train_tokens_with_labels = []
    for example in train_dataset:
      train_tokens_with_labels.append(example["label"])
      train_tokens_with_labels.extend(example["input_ids"])
    train_data = np.array(train_tokens_with_labels, dtype=np.uint16)

    val_tokens_with_labels = []
    for example in val_dataset:
      val_tokens_with_labels.append(example["label"])
      val_tokens_with_labels.extend(example["input_ids"])
    val_data = np.array(val_tokens_with_labels, dtype=np.uint16)

    metadata = {
      "vocab_size": len(self.tokenizer),
      "train_size": len(train_dataset),
      "val_size": len(val_dataset),
      "max_length": 1024,
    }

    return [train_data, val_data, metadata]

  def get_batch(self, split):
    if split == 'train':
        data = np.memmap('./data/sst2/train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('./data/sst2/val.bin', dtype=np.uint16, mode='r')

    num_examples = len(data) // (block_size + 1)
    sample_indecies = torch.randint(num_examples, (self.batch_size,))

    labels = []
    token_sequences = []

    for idx in sample_indecies:
        start_idx = idx * (block_size + 1)

        label = data[start_idx]
        token_sequence = data[start_idx + 1:start_idx + block_size + 1]

        labels.append(label)
        token_sequences.append(torch.from_numpy(token_sequence.astype(np.int64)))

    y = torch.tensor(labels, dtype=torch.long)
    x = torch.stack(token_sequences)

    if self.device == 'cuda':
        x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
    else:
        x, y = x.to(self.device), y.to(self.device)
    
    return x, y
  
  @torch.no_grad()
  def estimate_loss(self, ctx, eval_iters):
    out = {}
    self.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        all_preds = []
        all_targets = []
        
        for k in range(eval_iters):
            X, Y = self.get_batch(split)
            with ctx:
                logits, loss = self(X, Y)
                preds = torch.argmax(logits, dim=1)
            
            losses[k] = loss.item()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(Y.cpu().numpy())
        
        # Concatenate all predictions and targets
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Calculate F2 score
        f2 = fbeta_score(all_targets, all_preds, beta=2, average='binary', zero_division=0)
        
        # Calculate accuracy
        accuracy = accuracy_score(all_targets, all_preds)
        
        # Store metrics
        out[split] = losses.mean()
        out[f'{split}_f2'] = f2
        out[f'{split}_accuracy'] = accuracy
    
    self.train()
    return out

  def forward(self, tokens, targets=None):
    if self.pretrained_model is None:
        raise ValueError("Pretrained model not set. Call set_pretrained_model() first.")
    
    outputs = self.pretrained_model(tokens)

    # Use mean pooling over sequence length
    x = outputs.mean(dim=1)  # Shape: [batch_size, n_embd]

    normalized = self.layer_norm(x)
    logits = self.classifier(normalized)
        
    # Compute loss if targets provided
    if targets is not None:
        if targets.max() >= self.num_classes:
            targets = torch.clamp(targets, 0, self.num_classes-1)
        loss = F.cross_entropy(logits, targets)
    else:
        loss = None
    
    return logits, loss

