import pathlib

import numpy as np
from sklearn.metrics import fbeta_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast


class Task(nn.Module):
    def __init__(self, device: str,
                 tokenizer_vocab: pathlib.Path, tokenizer_merges: pathlib.Path,
                 data_prefix: pathlib.Path,
                 embedding_size: int, dropout: float, n_classes: int,
                 context_window: int,
                 input_feat: str, output_feat: str):
        super().__init__()
        self.pretrained_model = None
        self.device = device
        self.tokenizer_vocab = tokenizer_vocab
        self.tokenizer_merges = tokenizer_merges
        self.data_prefix = data_prefix

        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.dropout = dropout
        self.context_window = context_window

        self.input_feat = input_feat
        self.output_feat = output_feat

        ## 1e-5, 2e-5, 3e-5
        self.learning_rate = 3e-5
        self.weight_decay = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.grad_clip = .1
        self.min_lr = .1 * self.learning_rate
        # 16, 32
        self.batch_size = 16
        self.warmup_iter_ratio = .06
        self.lr_decay_iter_ratio = .9
        self.warmup_iters = None
        self.lr_decay_iters = None

        self.tokenizer = GPT2TokenizerFast(
            vocab_file=str(self.tokenizer_vocab),
            # f"{self.storage_prefix}/tokenizers/bpe-normal-number-preservation-vocab.json",
            merges_file=str(self.tokenizer_merges),
            # f"{self.storage_prefix}/tokenizers/bpe-normal-number-preservation-merges.txt",
            pad_token="<pad>",
            model_max_length=1024,
            padding_side="right",
            truncation=True,
            truncation_side="right"
        )

        self.layer_norm = nn.LayerNorm(self.embedding_size)

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_size, self.num_classes, bias=False)
        )

        with torch.no_grad():
            self.classifier[1].weight.data.normal_(mean=0.0, std=0.02)

        for pn, p in self.classifier.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / np.sqrt(2 * self.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def load_data(self, split: str):
        return np.memmap(self.data_prefix / ('train.bin' if split == 'train' else 'val.bin'), dtype=np.uint16, mode='r')

    def preprocess(self, examples):
        result = self.tokenizer(
            examples[self.input_feat],
            padding="max_length",
            truncation=True,
            max_length=1024
        )
        result["labels"] = examples[self.output_feat]
        return result

    def prepare_data(self, train_dataset, val_dataset):
        train_tokens_with_labels = []
        for example in train_dataset:
            train_tokens_with_labels.append(example[self.output_feat])
            train_tokens_with_labels.extend(example["input_ids"])
        train_data = np.array(train_tokens_with_labels, dtype=np.uint16)

        val_tokens_with_labels = []
        for example in val_dataset:
            val_tokens_with_labels.append(example[self.output_feat])
            val_tokens_with_labels.extend(example["input_ids"])
        val_data = np.array(val_tokens_with_labels, dtype=np.uint16)

        metadata = {
            "vocab_size": len(self.tokenizer),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "max_length": 1024,
        }

        return [train_data, val_data, metadata]

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
                targets = torch.clamp(targets, 0, self.num_classes - 1)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def get_batch(self, split):
        data = self.load_data(split)

        num_examples = len(data) // (self.context_window + 1)
        sample_indecies = torch.randint(num_examples, (self.batch_size,))

        labels = []
        token_sequences = []

        for idx in sample_indecies:
            start_idx = idx * (self.context_window + 1)

            label = data[start_idx]
            token_sequence = data[start_idx + 1:start_idx + self.context_window + 1]

            labels.append(label)
            token_sequences.append(torch.from_numpy(token_sequence.astype(np.int64)))

        y = torch.tensor(labels, dtype=torch.long)
        x = torch.stack(token_sequences)

        if self.device == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)

        return x, y
