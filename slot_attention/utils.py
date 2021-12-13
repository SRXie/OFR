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

def compute_corr_coef(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return cost

def compute_cos_distance(x: Tensor):
    """
    Tensor x should have shape (batch_size, num_slot, emb_size)
    """
    x_norm = torch.norm(x, p=2, dim=-1).detach()
    x_normed = x.div(x_norm.unsqueeze(-1).repeat(1,1,x.shape[-1]))
    # https://stats.stackexchange.com/questions/146221/is-cosine-similarity-identical-to-l2-normalized-euclidean-distance
    cos_distance = torch.cdist(x_normed, x_normed, p=2)/2
    return cos_distance

def compute_rank_correlation(x: Tensor, y: Tensor):
    """
    Function that measures Spearman’s correlation coefficient between target logits and output logits:
    https://www.programmersought.com/article/94895532714/
    """
    def _rank_correlation_(x, y):
        n = torch.tensor(x.shape[1])
        upper = 6 * torch.sum((y - x).pow(2), dim=1)
        down = n * (n.pow(2) - 1.0)
        return (1.0 - (upper / down)).mean(dim=-1).reshape(1)

    x = x.sort(dim=1)[1]
    y = y.sort(dim=1)[1]
    correlation = _rank_correlation_(x.float(), y.float())
    return correlation

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

def interleave_stack(x, y):
    return torch.stack((x, y), dim=1).view(2*x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])

def batched_index_select(input, dim, index):
    """
    https://discuss.pytorch.org/t/batched-index-select/9115/8
    """
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)

def compute_aggregated_loss(cat_slots, losses):
    slots_A, slots_B, slots_C, slots_D = torch.split(cat_slots, cat_slots.shape[0]//4, 0)
    batch_size, num_slots, slot_size = slots_A.shape

    ext_A = slots_A.sum(1)
    ext_B = slots_B.sum(1)
    ext_C = slots_C.sum(1)
    ext_D = slots_D.sum(1)
    loss = torch.norm(ext_A-ext_B+ext_C-ext_D, 2, -1)
    norm_term = torch.stack([torch.norm(ext_A-ext_B, 2, -1), torch.norm(ext_A-ext_D, 2, -1), torch.norm(ext_C-ext_B, 2, -1), torch.norm(ext_C-ext_D, 2, -1)], dim=-1)
    norm_term = torch.max(norm_term, dim=-1)[0]
    loss = loss.div(norm_term+0.0001)

    losses.append(loss)

def compute_pseudo_greedy_loss(cat_slots, losses, easy_neg=False, cos_sim=False):
    slots_A, slots_B, slots_C, slots_D = torch.split(cat_slots, cat_slots.shape[0]//4, 0)
    batch_size, num_slots, slot_size = slots_A.shape
    indices_A = torch.arange(0, num_slots, dtype=int).unsqueeze(0).repeat(batch_size, 1).to(cat_slots.device)

    greedy_criterion_AB = torch.norm(slots_A.view(batch_size, num_slots, 1, slot_size)-slots_B.view(batch_size, 1, num_slots, slot_size), 2, -1)
    _, indices_B = greedy_criterion_AB.min(-1)
    indices_B = indices_B.view(batch_size, 1)
    slots_B = batched_index_select(slots_B, 1, indices_B)

    greedy_criterion_AD = torch.norm(slots_A.view(batch_size, num_slots, 1, slot_size)-slots_D.view(batch_size, 1, num_slots, slot_size), 2, -1)
    _, indices_D = greedy_criterion_AD.min(-1)
    indices_D = indices_D.view(batch_size, 1)
    slots_D = batched_index_select(slots_D, 1, indices_D)

    greedy_criterion_DC = torch.norm(slots_D.view(batch_size, num_slots, 1, slot_size)-slots_C.view(batch_size, 1, num_slots, slot_size), 2, -1)
    _, indices_C = greedy_criterion_DC.min(-1)
    indices_C = indices_C.view(batch_size, 1)
    slots_C = batched_index_select(slots_C, 1, indices_C)

    losses.append(torch.norm(slots_A-slots_B+slots_C+slots_D, 2, -1))

    return torch.cat([indices_A, indices_B, indices_C, indices_D], 0)

    # # greedy assignment regardless of re-assignment
    # # TODO: check if there is a trivial solution to this assignment
    # ext_A = slots_A.view(batch_size, num_slots, 1, 1, 1, slot_size).expand(-1, -1, num_slots, num_slots, num_slots, -1)
    # ext_B = slots_B.view(batch_size, 1, num_slots, 1, 1, slot_size).expand(-1, num_slots, -1, num_slots, num_slots, -1)
    # ext_C = slots_C.view(batch_size, 1, 1, num_slots, 1, slot_size).expand(-1, num_slots, num_slots, -1, num_slots, -1)
    # ext_D = slots_D.view(batch_size, 1, 1, 1, num_slots, slot_size).expand(-1, num_slots, num_slots, num_slots, -1, -1)
    # if not cos_sim:
    #     greedy_criterion = torch.norm(ext_A-ext_B+ext_C-ext_D, 2, -1)
    #     norm_term = torch.stack([torch.norm(ext_A-ext_B, 2, -1), torch.norm(ext_A-ext_D, 2, -1), torch.norm(ext_C-ext_B, 2, -1), torch.norm(ext_C-ext_D, 2, -1)], dim=-1)
    #     norm_term = torch.max(norm_term, dim=-1)[0]
    #     greedy_criterion = greedy_criterion.div(norm_term+0.0001)
    # else:
    #     vector_a = (ext_A-ext_B).div(torch.norm(ext_A-ext_B, 2, -1).unsqueeze(-1).repeat(1,1,1,1,1,slot_size)+0.0001)
    #     vector_b = (ext_D-ext_C).div(torch.norm(ext_D-ext_C, 2, -1).unsqueeze(-1).repeat(1,1,1,1,1,slot_size)+0.0001)
    #     greedy_criterion = torch.norm(vector_a-vector_b, 2, -1)/2
    # # backtrace for greedy matching (3 times)
    # greedy_criterion, _ = greedy_criterion.min(-1)
    # greedy_criterion, _ = greedy_criterion.min(-1)
    # greedy_criterion, _ = greedy_criterion.min(-1)

    # greedy_loss = greedy_criterion.sum(dim=-1)/num_slots
    # losses.append(greedy_loss)

def compute_greedy_loss(cat_slots, losses, cos_sim=False):
    slots_A, slots_B, slots_C, slots_D = torch.split(cat_slots, cat_slots.shape[0]//4, 0)
    batch_size, num_slots, slot_size = slots_A.shape
    # greedy assignment without multi-assignment
    greedy_loss = torch.zeros(batch_size).to(cat_slots.device)
    cat_indices_holder = torch.arange(0, num_slots, dtype=int).unsqueeze(0).repeat(4*batch_size, 1).to(cat_slots.device)

    for i in range(num_slots):
        ext_A = slots_A.view(batch_size, num_slots-i, 1, 1, 1, slot_size).expand(-1, -1, num_slots-i, num_slots-i, num_slots-i, -1)
        ext_B = slots_B.view(batch_size, 1, num_slots-i, 1, 1, slot_size).expand(-1, num_slots-i, -1, num_slots-i, num_slots-i, -1)
        ext_C = slots_C.view(batch_size, 1, 1, num_slots-i, 1, slot_size).expand(-1, num_slots-i, num_slots-i, -1, num_slots-i, -1)
        ext_D = slots_D.view(batch_size, 1, 1, 1, num_slots-i, slot_size).expand(-1, num_slots-i, num_slots-i, num_slots-i, -1, -1)
        if not cos_sim:
            greedy_criterion = torch.norm(ext_A-ext_B+ext_C-ext_D, 2, -1)
            # norm_term = torch.stack([torch.norm(ext_A-ext_B, 2, -1), torch.norm(ext_A-ext_D, 2, -1), torch.norm(ext_C-ext_B, 2, -1), torch.norm(ext_C-ext_D, 2, -1)], dim=-1)
            # norm_term = torch.max(norm_term, dim=-1)[0]
            # greedy_criterion = greedy_criterion.div(norm_term+0.0001)
        else:
            vector_a = (ext_A-ext_B+ext_C).div(torch.norm(ext_A-ext_B+ext_C, 2, -1).unsqueeze(-1).repeat(1,1,1,1,1,slot_size)+0.0001)
            vector_b = (ext_D).div(torch.norm(ext_D, 2, -1).unsqueeze(-1).repeat(1,1,1,1,1,slot_size)+0.0001)
            greedy_criterion = torch.norm(vector_a-vector_b, 2, -1)/2
        # backtrace for greedy matching (3 times)
        greedy_criterion, indices_D = greedy_criterion.min(-1)
        greedy_criterion, indices_C = greedy_criterion.min(-1)
        greedy_criterion, indices_B = greedy_criterion.min(-1)
        greedy_criterion, indices_A = greedy_criterion.min(-1)
        greedy_loss+=greedy_criterion

        # print(indices_A[1])
        # print(indices_B[1, indices_A[1]])
        # print(indices_C[1 , indices_A[1], indices_B[1, indices_A[1]]])
        # print(indices_D[1, indices_A[1], indices_B[1, indices_A[1]], indices_C[1 , indices_A[1], indices_B[1, indices_A[1]]]])

        index_A = indices_A.view(indices_A.shape[0],1)

        index_B = batched_index_select(indices_B, 1, index_A)
        index_B = index_B.view(index_B.shape[0],1)

        index_C = batched_index_select(indices_C, 1, index_A)
        index_C = batched_index_select(index_C, 2, index_B)
        index_C = index_C.view(index_C.shape[0],1)

        index_D = batched_index_select(indices_D, 1, index_A)
        index_D = batched_index_select(index_D, 2, index_B)
        index_D = batched_index_select(index_D, 3, index_C)
        index_D = index_D.view(index_D.shape[0],1)

        replace = torch.zeros(batch_size*4, num_slots-i, dtype=torch.bool)
        replace = replace.to(cat_slots.device)
        index_cat = torch.cat([index_A, index_B, index_C, index_D], dim=0)
        slots_cat = torch.cat([slots_A, slots_B, slots_C, slots_D], dim=0)

        replace = replace.scatter(1, index_cat, True)
        # batched element swap
        tmp = batched_index_select(cat_indices_holder, 1, index_cat.squeeze(1)).squeeze(1).clone()
        cat_indices_holder[:, 0:num_slots-i-1] = torch.where(replace[:, 0:num_slots-i-1], cat_indices_holder[:, num_slots-i-1].unsqueeze(1).repeat(1, num_slots-i-1), cat_indices_holder[:, 0:num_slots-i-1])
        cat_indices_holder[:, num_slots-i-1] = tmp
        replace = replace.unsqueeze(-1).repeat(1, 1, slot_size)
        slots_cat = torch.where(replace, slots_cat[:,-1,:].unsqueeze(1).repeat(1, num_slots-i, 1), slots_cat)[:,:-1,:]
        slots_A, slots_B, slots_C, slots_D = torch.split(slots_cat, batch_size, 0)

    # greedy_loss = greedy_loss/num_slots
    losses.append(greedy_loss)
    return cat_indices_holder

def compute_cosine_loss(cat_slots_sorted, losses):
    slots_A, slots_B, slots_C, slots_D = torch.split(cat_slots_sorted, cat_slots_sorted.shape[0]//4, 0)
    vector_AB = (slots_A - slots_B).div(torch.norm(slots_A - slots_B, 2, -1).unsqueeze(-1).repeat(1, 1, cat_slots.shape[2])+0.0001)
    vector_DC = (slots_D - slots_C).div(torch.norm(slots_D - slots_C, 2, -1).unsqueeze(-1).repeat(1, 1, cat_slots.shape[2])+0.0001)
    cos_loss = torch.norm(vector_AB-vector_DC, 2, -1)/2
    losses.append(cos_loss.mean(1))

def compute_partition_loss(cat_slots_sorted, A_losses, D_losses, cos_sim=False):
    slots_A, slots_B, slots_C, slots_D = torch.split(cat_slots_sorted, cat_slots_sorted.shape[0]//4, 0)
    batch_size, num_slots, slot_size = slots_A.shape
    if not cos_sim:
        slots_A_delta = slots_A.view(batch_size, 1, num_slots, slot_size) - slots_A.view(1, batch_size, num_slots, slot_size)
        A_loss = (torch.norm(slots_A_delta, 2, -1).sum(2))#+torch.exp(-torch.norm(slots_A-slots_B, 2, -1).sum(1))
    else:
        unit_slots_A = slots_A.div(torch.norm(slots_A, 2, -1).unsqueeze(-1).repeat(1,1,slot_size)+0.0001).view(batch_size, 1, num_slots, slot_size)
        unit_slots_B = slots_B.div(torch.norm(slots_B, 2, -1).unsqueeze(-1).repeat(1,1,slot_size)+0.0001).view(batch_size, 1, num_slots, slot_size)
        unit_slots_D = slots_D.div(torch.norm(slots_D, 2, -1).unsqueeze(-1).repeat(1,1,slot_size)+0.0001).view(1, batch_size, num_slots, slot_size)
        A_loss = torch.exp(-torch.norm(unit_slots_A-unit_slots_D, 2, -1).sum(2)/2).sum(1)#+torch.exp(-torch.norm(unit_slots_A-unit_slots_B, 2, -1).squeeze(1).sum(1))
    A_losses.append(-A_loss)
    slots_D_prime = slots_A-slots_B+slots_C
    if not cos_sim:
        slots_D_delta = slots_D_prime.view(batch_size, 1, num_slots, slot_size) - slots_D.view(1, batch_size, num_slots, slot_size)
        D_loss = (torch.norm(slots_D_delta, 2, -1).sum(2))
    else:
        unit_slots_D_prime = slots_D_prime.div(torch.norm(slots_D_prime, 2, -1).unsqueeze(-1).repeat(1,1,slot_size)+0.0001)
        D_loss = torch.exp(-torch.norm(unit_slots_D_prime.view(batch_size, 1, num_slots, slot_size) - unit_slots_D.view(1, batch_size, num_slots, slot_size), 2, -1).sum(1)/2).sum(1)
    D_losses.append(-D_loss)

def bipartite_greedy_loss(cat_slots_sorted, slots_E, slots_F, losses_AE, losses_DF, cos_sim=False):
    slots_A, slots_B, slots_C, slots_D = torch.split(cat_slots_sorted, cat_slots_sorted.shape[0]//4, 0)
    slots_D_prime = slots_A-slots_B+slots_C
    batch_size, num_slots, slot_size = slots_A.shape
    # greedy assignment without multi-assignment
    greedy_loss_AE = torch.zeros(batch_size).to(cat_slots_sorted.device)
    greedy_loss_DF = torch.zeros(batch_size).to(cat_slots_sorted.device)
    cat_indices_holder = torch.arange(0, num_slots, dtype=int).unsqueeze(0).repeat(4*batch_size, 1).to(cat_slots_sorted.device)

    for i in range(num_slots):
        slots_AD = torch.cat([slots_A, slots_D_prime], 0)
        slots_EF = torch.cat([slots_E, slots_F], 0)
        ext_AD = slots_AD.view(2*batch_size, num_slots-i, 1, slot_size).expand(-1, -1, num_slots-i, -1)
        ext_EF = slots_EF.view(2*batch_size, 1, num_slots-i, slot_size).expand(-1, num_slots-i, -1, -1)

        if not cos_sim:
            greedy_criterion = torch.norm(ext_AD-ext_EF, 2, -1)
            # norm_term = torch.stack([torch.norm(ext_A-ext_B, 2, -1), torch.norm(ext_A-ext_D, 2, -1), torch.norm(ext_C-ext_B, 2, -1), torch.norm(ext_C-ext_D, 2, -1)], dim=-1)
            # norm_term = torch.max(norm_term, dim=-1)[0]
            # greedy_criterion = greedy_criterion.div(norm_term+0.0001)
        else:
            unit_vector_a = (ext_AD).div(torch.norm(ext_AD, 2, -1).unsqueeze(-1).repeat(1,1,1,slot_size)+0.0001)
            unit_vector_b = (ext_EF).div(torch.norm(ext_EF, 2, -1).unsqueeze(-1).repeat(1,1,1,slot_size)+0.0001)
            greedy_criterion = torch.norm(unit_vector_a-unit_vector_b, 2, -1)/2

        # backtrace for greedy matching (3 times)
        greedy_criterion, indices_EF = greedy_criterion.min(-1)
        greedy_criterion, indices_AD = greedy_criterion.min(-1)
        greedy_criterion_AE, greedy_criterion_DF = torch.split(greedy_criterion, batch_size, 0)
        greedy_loss_AE += greedy_criterion_AE
        greedy_loss_DF += greedy_criterion_DF

        index_AD = indices_AD.view(indices_AD.shape[0],1)
        index_EF = batched_index_select(indices_EF, 1, index_AD)
        index_EF = index_EF.view(index_EF.shape[0],1)

        replace = torch.zeros(batch_size*4, num_slots-i, dtype=torch.bool)
        replace = replace.to(cat_slots_sorted.device)
        index_cat = torch.cat([index_AD, index_EF], dim=0)
        slots_cat = torch.cat([slots_AD, slots_EF], dim=0)

        replace = replace.scatter(1, index_cat, True)
        # batched element swap
        tmp = batched_index_select(cat_indices_holder, 1, index_cat.squeeze(1)).squeeze(1).clone()
        cat_indices_holder[:, 0:num_slots-i-1] = torch.where(replace[:, 0:num_slots-i-1], cat_indices_holder[:, num_slots-i-1].unsqueeze(1).repeat(1, num_slots-i-1), cat_indices_holder[:, 0:num_slots-i-1])
        cat_indices_holder[:, num_slots-i-1] = tmp
        replace = replace.unsqueeze(-1).repeat(1, 1, slot_size)
        slots_cat = torch.where(replace, slots_cat[:,-1,:].unsqueeze(1).repeat(1, num_slots-i, 1), slots_cat)[:,:-1,:]
        slots_A, slots_D_prime, slots_E, slots_F = torch.split(slots_cat, batch_size, 0)

    losses_AE.append(-greedy_loss_AE)
    losses_DF.append(-greedy_loss_DF)

def compute_ari(table):
    """
    https://github.com/zhixuan-lin/IODINE/blob/3e8a74673c193d990832ca676f02f2d4956ef1d1/lib/utils/ari.py
    Compute ari, given the index table
    :param table: (r, s)
    :return:
    """

    # (r,)
    a = table.sum(axis=1)
    # (s,)
    b = table.sum(axis=0)
    n = a.sum()

    comb_a = comb(a, 2).sum()
    comb_b = comb(b, 2).sum()
    comb_n = comb(n, 2)
    comb_table = comb(table, 2).sum()

    if (comb_b == comb_a == comb_n == comb_table):
        # the perfect case
        ari = 1.0
    else:
        ari = (
            (comb_table - comb_a * comb_b / comb_n) /
            (0.5 * (comb_a + comb_b) - (comb_a * comb_b) / comb_n)
        )

    return ari


def compute_mask_ari(mask0, mask1):
    """
    Given two sets of masks, compute ari
    :param mask0: ground truth mask, (N0, H, W)
    :param mask1: predicted mask, (N1, H, W)
    :return:
    """

    # will first need to compute a table of shape (N0, N1)
    # (N0, 1, H, W)
    mask0 = mask0[:, None].byte()
    # (1, N1, H, W)
    mask1 = mask1[None, :].byte()
    # (N0, N1, H, W)
    agree = mask0 & mask1
    # (N0, N1)
    table = agree.sum(dim=-1).sum(dim=-1)

    return compute_ari(table.numpy())

def swap_bg_slot_back(cat_attns):#, cat_slots, cat_slots_nodup):
    batch_size, _, num_slots = cat_attns.shape
    batch_size = batch_size // 4
    cat_indices_holder = torch.arange(0, num_slots, dtype=int).unsqueeze(0).repeat(4*batch_size, 1).to(cat_attns.device)
    # get slot index with largest non-zero attention area
    cat_attns = cat_attns.permute(0,2,1)
    attns_area = torch.sum(cat_attns > 2.0/num_slots, dim=-1)

    cat_index = torch.argmax(attns_area, dim=1)
    # batched element swap
    replace = torch.zeros(batch_size*4, num_slots, dtype=torch.bool)
    replace = replace.to(cat_attns.device)
    replace = replace.scatter(1, cat_index.unsqueeze(1), True)

    tmp_index = batched_index_select(cat_indices_holder, 1, cat_index).squeeze(1).clone()
    cat_indices_holder[:, 0:num_slots-1] = torch.where(replace[:, 0:num_slots-1], cat_indices_holder[:, num_slots-1].unsqueeze(1).repeat(1, num_slots-1), cat_indices_holder[:, 0:num_slots-1])
    cat_indices_holder[:, num_slots-1] = tmp_index

    return cat_indices_holder

def captioned_masked_recons(recons, masks, slots, attns):
    cos_dis_pixel = compute_cos_distance(attns.permute(0,2,1)) # to have shape (batch_size, num_slot, emb_size)
    pixel_dup_sim, pixel_dup_idx = torch.sort(cos_dis_pixel, dim=-1)
    pixel_dup_sim = pixel_dup_sim[:,:,1]
    pixel_dup_idx = pixel_dup_idx[:,:,1]

    cos_dis_feature = compute_cos_distance(slots)
    feature_dup_sim, feature_dup_idx = torch.sort(cos_dis_feature, dim=-1)
    feature_dup_sim = feature_dup_sim[:,:,1]
    feature_dup_idx = feature_dup_idx[:,:,1]

    attn = attns.permute(0, 2, 1).view(recons.shape[0], recons.shape[1], recons.shape[3], recons.shape[4])
    masked_recons = recons #* masks + (1 - masks)
    masked_recons = to_rgb_from_tensor(masked_recons)
    masked_attns = torch.zeros_like(recons)
    masked_attns[:,:,2,:,:] = masked_attns[:,:,2,:,:]+masks.squeeze(2)
    masked_attns[:,:,0,:,:] = masked_attns[:,:,0,:,:]+attn
    masked_attns = to_rgb_from_tensor(masked_attns)

    # TODO: We keep this mass instead of attn mass for now
    mask_mass = masks.view(recons.shape[0], recons.shape[1], -1)
    mask_mass = torch.where(mask_mass>=mask_mass.max(dim=1)[0].unsqueeze(1).repeat(1,recons.shape[1],1), mask_mass, torch.zeros_like(mask_mass)).sum(-1)
    masks = masks.repeat(1,1,3,1,1)
    masks = to_rgb_from_tensor(masks)
    for i in range(masked_recons.shape[0]):
        for j in range(masked_recons.shape[1]):
            img = transforms.ToPILImage()(masked_recons[i,j])
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
            feature_text = "feat: "+str(feature_dup_idx[i,j].item())+" - {:.4f}".format(feature_dup_sim[i,j].item())
            draw.text((4,55), feature_text, (0, 0, 0), font=font)
            img = transforms.ToTensor()(img)
            img = to_tensor_from_rgb(img)
            masked_recons[i,j] = img
        for j in range(masked_recons.shape[1]):
            img = transforms.ToPILImage()(masked_attns[i,j])
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
            pixel_text = "attn: "+str(pixel_dup_idx[i,j].item())+" - {:.4f}".format(pixel_dup_sim[i,j].item())
            draw.text((4,0), pixel_text, (0, 0, 0), font=font)
            img = transforms.ToTensor()(img)
            img = to_tensor_from_rgb(img)
            masked_attns[i,j] = img
        for j in range(masked_recons.shape[1]):
            img = transforms.ToPILImage()(masks[i,j])
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
            pixel_text = "attn: {:.4f}".format(mask_mass[i,j].item())
            draw.text((4,0), pixel_text, (0, 0, 0), font=font)
            img = transforms.ToTensor()(img)
            img = to_tensor_from_rgb(img)
            masks[i,j] = img
    return masked_recons, masked_attns, masks
