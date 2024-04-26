from dataclasses import dataclass
from typing import Tuple

@dataclass
class MAMLConfigADIMA:
    random_seed: int = 42
    meta_lr: float = 1e-4  # "beta"
    num_epochs: int = 400
    inner_training_steps: int = 1    # In MAML, generally just 1 gradient descent step in per-task "inner" training
    batch_size: int = 4  # "K" (like K-shot); symbolic of MAML training loop optimizing K-shot per-task; NOTE: affects VRAM
    inner_lr: float = 1e-4     # "alpha"
    n_tasks: int = 3     # Number of tasks to sample from distribution of tasks; NOTE: affects VRAM
