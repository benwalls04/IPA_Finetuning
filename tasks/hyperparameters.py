class Hyperparameters:
    def __init__(self):
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

    def override_settings(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, type(getattr(self, key))(value))
