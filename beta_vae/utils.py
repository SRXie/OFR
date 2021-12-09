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

def conv_transpose_out_shape(in_size, stride, padding, kernel_size, out_padding, dilation=1):
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1

def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"

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

def compute_loss(cat_zs, losses):
    zs_A, zs_B, zs_C, zs_D = torch.split(cat_zs, cat_zs.shape[0]//4, 0)
    batch_size, z_dim = zs_A.shape

    loss = torch.norm(zs_A-zs_B+zs_C-zs_D, 2, -1)

    losses.append(loss)

def compute_partition_loss(cat_zs, A_losses, D_losses):
    zs_A, zs_B, zs_C, zs_D = torch.split(cat_zs, cat_zs.shape[0]//4, 0)
    batch_size, z_dim = zs_A.shape

    zs_A_delta = zs_A.view(batch_size, 1, z_dim) - zs_D.view(1, batch_size, z_dim)
    A_loss = -torch.norm(zs_A_delta, 2, -1)

    zs_D_prime = zs_A-zs_B+zs_C
    zs_D_delta = zs_D_prime.view(batch_size, 1, z_dim) - zs_D.view(1, batch_size, z_dim)
    D_loss = -torch.norm(zs_D_delta, 2, -1)

    A_losses.append(A_loss)
    D_losses.append(D_loss)

def compute_partition_loss_hard(cat_zs, zs_E, zs_F, AE_losses, DF_losses):
    zs_A, zs_B, zs_C, zs_D = torch.split(cat_zs, cat_zs.shape[0]//4, 0)
    batch_size, z_dim = zs_A.shape
    zs_D_prime = zs_A-zs_B+zs_C

    zs_AD = torch.cat([zs_A, zs_D_prime], 0)
    zs_EF = torch.cat([zs_E, zs_F], 0)

    zs_delta = zs_AD - zs_EF
    loss = -torch.norm(zs_delta, 2, -1)

    AE_loss, DF_loss = torch.split(loss, batch_size, 0)

    AE_losses.append(AE_loss)
    DF_losses.append(DF_loss)
