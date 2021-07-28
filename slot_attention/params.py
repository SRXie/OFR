from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SlotAttentionParams:
    seed = 3
    gpu_id = '0'
    lr: float = 0.0004
    batch_size: int = 64
    val_batch_size: int = 64
    resolution: Tuple[int, int] = (64, 64)
    num_slots: int = 7
    num_iterations: int = 4
    data_root: str = "/datasets01/CLEVR_v1.0/060817/"
    test_root: str = "/checkpoint/siruixie/clevr_obj_test/output/"
    gpus: int = 1
    max_epochs: int = 100
    num_sanity_val_steps: int = 1
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None
    num_test_images: Optional[int] = 1
    empty_cache: bool = True
    is_logger_enabled: bool = True
    is_verbose: bool = True
    num_workers: int = 4
    n_samples: int = 5
    warmup_steps_pct: float = 0.02
    decay_steps_pct: float = 0.2
