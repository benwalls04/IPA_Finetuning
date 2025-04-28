import pathlib

from tasks.common import Task

class ClassificationCOLA(Task):
  def __init__(self, device,
               tokenizer_vocab: pathlib.Path, tokenizer_merges: pathlib.Path,
               data_prefix: pathlib.Path,
               num_embed: int = 768, dropout: float = 0.1, n_classes: int = 2,
               context_size: int = 1024,
               input_feat: str = 'sentence', output_feat: str = 'label',
               ipa: bool = False):
    super().__init__(
        'cola',
        device, tokenizer_vocab, tokenizer_merges, data_prefix,
        num_embed, dropout, n_classes, context_size,
        input_feat, output_feat, ipa
    )