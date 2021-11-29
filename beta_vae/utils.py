from typing import Any
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch
import random
import numpy as np
from scipy.special import comb
from pytorch_lightning import Callback
from torchvision.transforms import transforms
from PIL import ImageDraw, ImageFont

import wandb

Tensor = TypeVar("torch.tensor")
T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


def conv_transpose_out_shape(in_size, stride, padding, kernel_size, out_padding, dilation=1):
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1


def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


def rescale(x: Tensor) -> Tensor:
    return x * 2 - 1


def compact(l: Any) -> Any:
    return list(filter(None, l))


def first(x):
    return next(iter(x))


def only(x):
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


class ImageLogCallback(Callback):
    def __init__(self):
        self.step = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images()
                trainer.logger.experiment.log({"images": [wandb.Image(images)]}, commit=False)
                # Uset this line for tensorboard
                # trainer.logger.experiment.add_images('eval_images', images, self.step, dataformats='CHW')
            self.step += 1


def to_rgb_from_tensor(x: Tensor):
    return (x * 0.5 + 0.5).clamp(0, 1)

def to_tensor_from_rgb(x: Tensor):
    return 2.0*(x - 0.5)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def split_and_interleave_stack(input, split_size):
    x, y, z, w = torch.split(input, split_size, 0)
    view_shape = [size for size in x.shape]
    view_shape[0]*=4
    return torch.stack((x, y, z, w), dim=1).view(view_shape)

def compute_loss(cat_zs, losses, easy_neg=False):
    zs_A, zs_B, zs_C, zs_D = torch.split(cat_zs, cat_zs.shape[0]//4, 0)
    batch_size, z_dim = zs_A.shape
    if easy_neg:
        zs_D = zs_D[torch.randperm(batch_size)]

    loss = torch.norm(zs_A-zs_B+zs_C-zs_D, 2, -1)
    norm_term = torch.stack([torch.norm(zs_A-zs_B, 2, -1), torch.norm(zs_A-zs_D, 2, -1), torch.norm(zs_C-zs_B, 2, -1), torch.norm(zs_C-zs_D, 2, -1)], dim=-1)
    norm_term = torch.max(norm_term, dim=-1)[0]
    loss = loss.div(norm_term+0.0001)

    losses.append(loss)
