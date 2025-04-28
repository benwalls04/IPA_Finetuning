import pathlib

from tasks.common import Task
from tasks.hyperparameters import Hyperparameters


class HyperparametersSST2(Hyperparameters):
    def __init__(self):
        super().__init__()
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


class ClassificationSST2(Task):
    def __init__(self, device,
               tokenizer_vocab: pathlib.Path, tokenizer_merges: pathlib.Path,
               data_prefix: pathlib.Path, hyperparameters: Hyperparameters = None,
               num_embed: int = 768, dropout: float = 0.1, n_classes: int = 2,
               context_size: int = 1024,
               input_feat: str = 'sentence', output_feat: str = 'label',
               ipa: bool = False):
        if hyperparameters is None:
            hyperparameters = HyperparametersSST2()
        super().__init__(
            'sst2',
            device, tokenizer_vocab, tokenizer_merges, data_prefix,
            num_embed, dropout, n_classes, context_size,
            input_feat, output_feat, hyperparameters, ipa
        )
