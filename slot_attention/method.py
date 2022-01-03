import pytorch_lightning as pl
import torch
import math
import numpy as np
from torch import optim
from torchvision import utils as vutils
from torchvision.transforms import transforms
from datetime import datetime


from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import Tensor
from slot_attention.utils import to_rgb_from_tensor, to_tensor_from_rgb
from slot_attention.utils import compute_cos_distance, compute_rank_correlation
from slot_attention.utils import batched_index_select
from slot_attention.utils import compute_greedy_loss, compute_cosine_loss
from slot_attention.utils import compute_shuffle_greedy_loss, compute_bipartite_greedy_loss
from slot_attention.utils import swap_bg_slot_back
from slot_attention.utils import captioned_masked_recons
from slot_attention.utils import split_and_interleave_stack
from slot_attention.utils import compute_corr_coef


class SlotAttentionMethod(pl.LightningModule):
    def __init__(self, model: SlotAttentionModel, datamodule: pl.LightningDataModule, params: SlotAttentionParams):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params
        self.save_hyperparameters('params')
        self.random_projection_init = False

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self.model.loss_function(batch)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss

    def sample_images(self):
        dl = self.datamodule.obj_test_dataloader()
        random_idx = torch.randint(high=len(dl), size=(1,))
        batch = next(iter(dl))  # list of A, B, C, D, E, F -- E and F are hard negatives
        perm = torch.randperm(self.params.val_batch_size)
        idx = perm[: self.params.n_samples]
        batch = torch.cat([b[idx] for b in batch[:4]], 0)

        if self.params.gpus > 0:
            batch = batch.to(self.device)

        with torch.no_grad():

            recon_combined, recons, masks, slots, attns, recon_combined_nodup, recons_nodup, masks_nodup, slots_nodup = self.model.forward(batch, dup_threshold=self.params.dup_threshold, algebra=True)

            # throw background slot back
            cat_indices = swap_bg_slot_back(attns)
            recons = batched_index_select(recons, 1, cat_indices)
            masks = batched_index_select(masks, 1, cat_indices)
            slots = batched_index_select(slots, 1, cat_indices)
            attns = batched_index_select(attns, 2, cat_indices)
            # recons_nodup = batched_index_select(recons_nodup, 1, cat_indices)
            # masks_nodup = batched_index_select(masks_nodup, 1, cat_indices)
            # slots_nodup = batched_index_select(slots_nodup, 1, cat_indices)

            # reorder with matching
            cat_indices = compute_greedy_loss(slots, [])
            recons_perm = batched_index_select(recons, 1, cat_indices)
            masks_perm = batched_index_select(masks, 1, cat_indices)
            slots_perm = batched_index_select(slots, 1, cat_indices)
            attns_perm = batched_index_select(attns, 2, cat_indices)
            masked_recons_perm, masked_attn_perm, recons_perm = captioned_masked_recons(recons_perm, masks_perm, slots_perm, attns_perm)

            # No need to match again
            # cat_indices_nodup = compute_greedy_loss(slots_nodup, [])
            recons_perm_nodup = recons_nodup #batched_index_select(recons_nodup, 1, cat_indices_nodup)
            masks_perm_nodup = masks_nodup #batched_index_select(masks_nodup, 1, cat_indices_nodup)
            slots_perm_nodup = slots_nodup #batched_index_select(slots_nodup, 1, cat_indices_nodup)
            attns_perm_nodup = attns #batched_index_select(attns, 2, cat_indices_nodup)
            masked_recons_perm_nodup, masked_attn_perm_nodup, recons_perm_nodup = captioned_masked_recons(recons_perm_nodup, masks_perm_nodup, slots_perm_nodup, attns_perm_nodup)

            batch = split_and_interleave_stack(batch, self.params.n_samples)
            recon_combined = split_and_interleave_stack(recon_combined, self.params.n_samples)
            recons_perm = split_and_interleave_stack(recons_perm, self.params.n_samples)
            masks_perm = split_and_interleave_stack(masks_perm, self.params.n_samples)
            slots_perm = split_and_interleave_stack(slots_perm, self.params.n_samples)
            masked_attn_perm = split_and_interleave_stack(masked_attn_perm, self.params.n_samples)
            recon_combined_nodup = split_and_interleave_stack(recon_combined_nodup, self.params.n_samples)
            recons_perm_nodup = split_and_interleave_stack(recons_perm_nodup, self.params.n_samples)
            masks_perm_nodup = split_and_interleave_stack(masks_perm_nodup, self.params.n_samples)
            slots_perm_nodup = split_and_interleave_stack(slots_perm_nodup, self.params.n_samples)
            masked_recons_perm = split_and_interleave_stack(masked_recons_perm, self.params.n_samples)
            masked_recons_perm_nodup = split_and_interleave_stack(masked_recons_perm_nodup, self.params.n_samples)
            masked_attn_perm_nodup = split_and_interleave_stack(masked_attn_perm_nodup, self.params.n_samples)

            # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
            out = to_rgb_from_tensor(
                torch.cat(
                    [
                        torch.cat([batch.unsqueeze(1), batch.unsqueeze(1)], dim=0),  # original images
                        torch.cat([recon_combined.unsqueeze(1),recon_combined_nodup.unsqueeze(1)], dim=0),  # reconstructions
                        torch.cat([masked_recons_perm, masked_recons_perm_nodup], dim=0),
                        torch.cat([recons_perm, recons_perm_nodup], dim=0),
                        torch.cat([masked_attn_perm, masked_attn_perm_nodup], dim=0),  # each slot
                    ],
                    dim=1,
                )
            )

            batch_size, num_slots, C, H, W = recons.shape
            images = vutils.make_grid(
                out.view(2 * batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
            )

        return images

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        # batch is a list of lengthn num_slots+1
        with torch.no_grad():
            val_loss = self.model.loss_function(batch[0], batch[1:-1], batch[-1])
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_ari_mask = np.stack([x["mask_ari"] for x in outputs]).mean()

        # Algebra Test starts here
        odl = self.datamodule.obj_test_dataloader()
        adl = self.datamodule.attr_test_dataloader()

        obj_greedy_losses_nodup, obj_greedy_std_nodup = [], []
        obj_greedy_losses_nodup_en_A, obj_greedy_losses_nodup_en_D, attr_greedy_losses_nodup_en = [], [], []
        # obj_greedy_losses_nodup_hn_A, obj_greedy_losses_nodup_hn_D, attr_greedy_losses_nodup_hn = [], [], []
        # obj_pd_greedy_losses, attr_pd_greedy_losses = [], []
        # obj_pd_greedy_losses_en_A, obj_pd_greedy_losses_en_D, attr_pd_greedy_losses_en = [], [], []
        # obj_pd_greedy_losses_hn_A, obj_pd_greedy_losses_hn_D, attr_pd_greedy_losses_hn = [], [], []

        # obj_greedy_cos_losses_nodup, attr_greedy_cos_losses_nodup = [], []
        # obj_greedy_cos_losses_nodup_en_A, obj_greedy_cos_losses_nodup_en_D, attr_greedy_cos_losses_nodup_en = [], [], []
        # obj_greedy_cos_losses_nodup_hn_A, obj_greedy_cos_losses_nodup_hn_D, attr_greedy_cos_losses_nodup_hn = [], [], []
        # obj_pd_greedy_cos_losses, attr_pd_greedy_cos_losses = [], []
        # obj_pd_greedy_cos_losses_en, attr_pd_greedy_cos_losses_en = [], []
        # obj_pd_greedy_cos_losses_hn, attr_pd_greedy_cos_losses_hn = [], []

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        z_norm, z_angle = [], []
        scalings, angles, scaling_deltas, angle_deltas, scaling_ratios, angle_ratios = [], [], [], [], [], []

        def compute_test_losses(dataloader, losses_nodup, losses_nodup_en_A, losses_nodup_en_D, std_nodup, dup_threshold=None):
            # b_prev = datetime.now()
            for batch in dataloader:
                # print("load data:", datetime.now()-b_prev)
                # sample_losses = []
                # batch is a length-4 list, each element is a tensor of shape (batch_size, 3, width, height)
                batch_size = batch[0].shape[0]
                cat_batch = torch.cat(batch[:4]+[batch[3]], 0)
                if self.params.gpus > 0:
                    cat_batch = cat_batch.to(self.device)
                cat_slots, cat_attns, cat_slots_nodup = self.model.forward(cat_batch, slots_only=True, dup_threshold=dup_threshold)

                # compute_bipartite_greedy_loss(cat_slots_nodup[3*batch_size:4*batch_size], cat_slots_nodup[4*batch_size:], std_nodup)
                # Here we compute the angle between DC and D'C
                cat_slots_two_fwd = torch.cat([cat_slots_nodup[3*batch_size:4*batch_size], cat_slots_nodup[2*batch_size:3*batch_size], cat_slots_nodup[2*batch_size:3*batch_size], cat_slots_nodup[4*batch_size:]], 0)
                cat_indices = compute_greedy_loss(cat_slots_two_fwd, std_nodup)
                cat_slots_two_fwd = batched_index_select(cat_slots_two_fwd, 1, cat_indices)
                CD_prime = (cat_slots_two_fwd[3*batch_size: 4*batch_size]-cat_slots_two_fwd[2*batch_size: 3*batch_size]).view(batch_size, -1)
                CD = (cat_slots_two_fwd[: batch_size]-cat_slots_two_fwd[1*batch_size: 2*batch_size]).view(batch_size, -1)
                CD_norm = torch.norm(CD, 2, -1)
                CD_prime_norm = torch.norm(CD_prime, 2, -1)
                # losses_nodup[-1]=torch.acos(torch.clamp((torch.square(slots_D_norm)+torch.square(slots_D_prime_norm)-losses_nodup[-1]).div(2*slots_D_norm*slots_D_prime_norm), max=1.0))
                slots_D = cat_slots_two_fwd[:batch_size].view(batch_size, -1)
                slots_D_prime = cat_slots_two_fwd[3*batch_size:].view(batch_size, -1)
                z_angle.append(torch.acos(torch.clamp(cos(slots_D, slots_D_prime), max=1.0)))
                std_nodup[-1]=torch.acos(torch.clamp((torch.square(CD_norm)+torch.square(CD_prime_norm)-std_nodup[-1]).div(2*CD_norm*CD_prime_norm), max=1.0))

                # slots_E = cat_slots[4*batch_size: 5*batch_size]
                # slots_E_nodup = cat_slots_nodup[4*batch_size: 5*batch_size]
                # slots_F = cat_slots[5*batch_size: 6*batch_size]
                # slots_F_nodup = cat_slots_nodup[5*batch_size: 6*batch_size]
                cat_slots = cat_slots[:4*batch_size]
                cat_attns = cat_attns[:4*batch_size]
                cat_slots_nodup = cat_slots_nodup[:4*batch_size]

                # cat_slots_nodup = torch.Tensor([[[1.0], [3.0], [2.0]], [[2.0], [0.5], [1.0]], [[0.5], [0.5], [2.0]], [[0.5], [0.5], [1.0]], [[0.5], [0.5], [6.0]], [[0.5], [4.0], [0.5]], [[2.0], [3.0], [6.0]], [[2.0], [4.0], [0.5]]]).to(cat_slots_nodup.device)
                # batch_size = 2
                cat_indices = compute_greedy_loss(cat_slots_nodup, losses_nodup)

                # starting here we want to use the cat_indices and match A' and D' to it A' = batch[4], D' = batch[5]
                # cat_batch_hn = torch.cat(batch[:3]+[batch[-1]], 0)
                # if self.params.gpus > 0:
                #     cat_batch_hn = cat_batch_hn.to(self.device)
                # cat_slots_hn, cat_attns_hn, cat_slots_nodup_hn = self.model.forward(cat_batch_hn, slots_only=True, dup_threshold=dup_threshold)

                # compute_cosine_loss(cat_slots_nodup, cat_indices, cos_losses_nodup)
                cat_slots_nodup_sorted = batched_index_select(cat_slots_nodup, 1, cat_indices)
                # print("cat_slots_nodup_sorted", cat_slots_nodup_sorted)
                slots_A, slots_B, slots_C, slots_D = torch.split(cat_slots_nodup_sorted.view(4*batch_size, -1), batch_size, 0)
                slots_A_norm = torch.norm(slots_A, 2, -1)
                slots_B_norm = torch.norm(slots_B, 2, -1)
                slots_C_norm = torch.norm(slots_C, 2, -1)
                slots_D_norm = torch.norm(slots_D, 2, -1)
                scaling_AB = slots_A_norm.div(slots_B_norm)
                scaling_DC = slots_D_norm.div(slots_C_norm)
                angle_AB = torch.acos(torch.clamp(cos(slots_A, slots_B), max=1.0))
                angle_DC = torch.acos(torch.clamp(cos(slots_D, slots_C), max=1.0))

                z_norm.append(torch.cat([slots_A_norm, slots_B_norm, slots_C_norm, slots_D_norm], 0))
                scalings.append(torch.cat([scaling_AB, scaling_DC], 0))
                angles.append(torch.cat([angle_AB, angle_DC], 0))
                scaling_deltas.append(torch.abs(scaling_AB - scaling_DC))
                angle_deltas.append(torch.abs(angle_AB - angle_DC))
                scaling_ratios.append(torch.maximum(scaling_AB, scaling_DC).div(torch.minimum(scaling_AB, scaling_DC)))
                angle_ratios.append(torch.maximum(angle_AB, angle_DC).div(torch.minimum(angle_AB, angle_DC)))
                #slots_D_prime_norm = torch.norm((cat_slots_nodup_sorted[: batch_size]-cat_slots_nodup_sorted[1*batch_size: 2*batch_size]+cat_slots_nodup_sorted[2*batch_size: 3*batch_size]).view(batch_size, -1), 2, -1)
                DC_norm = torch.norm(slots_D-slots_C, 2, -1)
                AB_norm = torch.norm((cat_slots_nodup_sorted[: batch_size]-cat_slots_nodup_sorted[1*batch_size: 2*batch_size]).view(batch_size, -1), 2, -1)
                #print("delta: ", ((torch.square(slots_D_norm)+torch.square(slots_D_prime_norm)-losses_nodup[-1]).div(2*slots_D_norm*slots_D_prime_norm)).max())
                # losses_nodup[-1]=torch.acos(torch.clamp((torch.square(slots_D_norm)+torch.square(slots_D_prime_norm)-losses_nodup[-1]).div(2*slots_D_norm*slots_D_prime_norm), max=1.0))
                losses_nodup[-1]=torch.acos(torch.clamp((torch.square(AB_norm)+torch.square(DC_norm)-losses_nodup[-1]).div(2*AB_norm*DC_norm), max=1.0))
                # print("losses_nodup", losses_nodup[-1])
                compute_shuffle_greedy_loss(cat_slots_nodup_sorted, losses_nodup_en_A, losses_nodup_en_D)
                # print("losses_nodup_en_D", losses_nodup_en_D)

                # bipartite_greedy_loss(cat_slots_nodup_sorted, slots_E_nodup, slots_F_nodup, losses_nodup_hn_A, losses_nodup_hn_D)
                # compute_greedy_loss(cat_slots_nodup_hn, losses_nodup_hn)

                # cat_indices = compute_pseudo_greedy_loss(cat_slots, pseudo_losses)
                # cat_slots_sorted = batched_index_select(cat_slots, 1, cat_indices)
                # compute_shuffle_greedy_loss(cat_slots_sorted, pseudo_losses_en_A, pseudo_losses_en_D)
                # bipartite_greedy_loss(cat_slots_sorted, slots_E, slots_F, pseudo_losses_hn_A, pseudo_losses_hn_D)
                # compute_pseudo_greedy_loss(cat_slots, pseudo_losses_en, easy_neg=True)
                # compute_pseudo_greedy_loss(cat_slots_hn, pseudo_losses_hn)

                # cat_indices = compute_greedy_loss(cat_slots_nodup, cos_losses_nodup, cos_sim=True)
                # compute_partition_loss(cat_slots_nodup, cat_indices, cos_losses_nodup_en_A, cos_losses_nodup_en_D, cos_sim=True)
                # bipartite_greedy_loss(cat_slots_nodup, cat_indices, slots_E_nodup, slots_F_nodup, cos_losses_nodup_hn_A, cos_losses_nodup_hn_D, cos_sim=True)
                # # compute_greedy_loss(cat_slots_nodup_hn, cos_losses_nodup_hn, cos_sim=True)

                # compute_pseudo_greedy_loss(cat_slots, pseudo_cos_losses, cos_sim=True)
                # compute_pseudo_greedy_loss(cat_slots, pseudo_cos_losses_en, easy_neg=True, cos_sim=True)
                # compute_pseudo_greedy_loss(cat_slots_hn, pseudo_cos_losses_hn, cos_sim=True)


        # with torch.no_grad():
        #     compute_test_losses(odl, obj_greedy_losses_nodup, obj_greedy_losses_nodup_en_A, obj_greedy_losses_nodup_en_D, obj_greedy_std_nodup, dup_threshold=self.params.dup_threshold)
        #     # compute_test_losses(adl, attr_pd_greedy_losses, attr_pd_greedy_losses_en, attr_pd_greedy_losses_hn, attr_greedy_losses_nodup, attr_greedy_losses_nodup_en, attr_greedy_losses_nodup_hn,
        #     #     attr_pd_greedy_cos_losses, attr_pd_greedy_cos_losses_en, attr_pd_greedy_cos_losses_hn, attr_greedy_cos_losses_nodup, attr_greedy_cos_losses_nodup_en, attr_greedy_cos_losses_nodup_hn, dup_threshold=self.params.dup_threshold)

        #     avg_z_norm = torch.cat(z_norm, 0).mean()
        #     avg_z_angle = torch.cat(z_angle, 0).mean()
        #     avg_scaling = torch.cat(scalings, 0).mean()
        #     avg_angle = torch.cat(angles, 0).mean()
        #     avg_scaling_delta = torch.cat(scaling_deltas, 0).mean()
        #     avg_angle_delta = torch.cat(angle_deltas, 0).mean()
        #     avg_scaling_ratio = torch.cat(scaling_ratios, 0).mean()
        #     avg_angle_ratio = torch.cat(angle_ratios, 0).mean()
        #     slot_std = torch.cat(obj_greedy_std_nodup, 0)
        #     obj_greedy_loss_nodup = torch.cat(obj_greedy_losses_nodup, 0)
        #     obj_greedy_loss_nodup_en_A = torch.cat([x for x in obj_greedy_losses_nodup_en_A], 0)
        #     obj_greedy_loss_nodup_en_D = torch.cat([x for x in obj_greedy_losses_nodup_en_D], 0)
        #     # obj_greedy_loss_nodup_hn_A = torch.cat(obj_greedy_losses_nodup_hn_A, 0)/(avg_slot_norm*self.model.num_slots) # this should be changed to log sum exp if there are more than one hard negatives
        #     # obj_greedy_loss_nodup_hn_D = torch.cat(obj_greedy_losses_nodup_hn_D, 0)/(avg_slot_norm*self.model.num_slots)
        #     # obj_greedy_loss_nodup_A = torch.cat([0.9*(x/(avg_slot_norm*self.model.num_slots)).mean(1)+0.1*y/(avg_slot_norm*self.model.num_slots) for x, y in zip(obj_greedy_losses_nodup_en_A, obj_greedy_losses_nodup_hn_A)], 0)
        #     # obj_greedy_loss_nodup_D = torch.cat([0.9*(x/(avg_slot_norm*self.model.num_slots)).mean(1)+0.1*y/(avg_slot_norm*self.model.num_slots) for x, y in zip(obj_greedy_losses_nodup_en_D, obj_greedy_losses_nodup_hn_D)], 0)
        #     avg_obj_greedy_ratio_en = ((obj_greedy_loss_nodup+obj_greedy_loss_nodup_en_D).div(obj_greedy_loss_nodup_en_D)).mean()
        #     # avg_obj_greedy_ratio_hn = ((obj_greedy_loss_nodup+obj_greedy_loss_nodup_hn_D).div(obj_greedy_loss_nodup_hn_D)).mean()
        #     # avg_obj_greedy_ratio = ((obj_greedy_loss_nodup+obj_greedy_loss_nodup_D).div(obj_greedy_loss_nodup_A)).mean()
        #     std_obj_greedy_loss = obj_greedy_loss_nodup.std()/math.sqrt(obj_greedy_loss_nodup.shape[0])
        #     avg_obj_greedy_loss = obj_greedy_loss_nodup.mean()
        #     avg_slot_std = slot_std.mean()
        #     avg_obj_greedy_ctrast_en = avg_obj_greedy_loss+obj_greedy_loss_nodup_en_D.mean()
        #     # avg_obj_greedy_ctrast_hn = avg_obj_greedy_loss+obj_greedy_loss_nodup_hn_D.mean()
        #     # avg_obj_greedy_ctrast = avg_obj_greedy_loss+obj_greedy_loss_nodup_D.mean()

        #     # avg_attr_greedy_loss_nodup = torch.cat(attr_greedy_losses_nodup, 0)
        #     # avg_attr_greedy_loss_nodup_en = (torch.cat(attr_greedy_losses_nodup_en, 0)-avg_attr_greedy_loss_nodup).mean()
        #     # avg_attr_greedy_loss_nodup_hn = (torch.cat(attr_greedy_losses_nodup_hn, 0)-avg_attr_greedy_loss_nodup).mean()
        #     # std_attr_greedy_loss_nodup = avg_attr_greedy_loss_nodup.std()/math.sqrt(avg_attr_greedy_loss_nodup.shape[0])
        #     # avg_attr_greedy_loss_nodup = avg_attr_greedy_loss_nodup.mean()

        #     # obj_pd_greedy_loss = torch.cat(obj_pd_greedy_losses, 0)/(avg_slot_norm*self.model.num_slots)
        #     # obj_pd_greedy_loss_en_A = torch.cat([(x/(avg_slot_norm*self.model.num_slots)).mean(1) for x in obj_pd_greedy_losses_en_A], 0)
        #     # obj_pd_greedy_loss_en_D = torch.cat([(x/(avg_slot_norm*self.model.num_slots)).mean(1) for x in obj_pd_greedy_losses_en_D], 0)
        #     # obj_pd_greedy_loss_hn_A = torch.cat(obj_pd_greedy_losses_hn_A, 0)/(avg_slot_norm*self.model.num_slots) # this should be changed to log sum exp if there are more than one hard negatives
        #     # obj_pd_greedy_loss_hn_D = torch.cat(obj_pd_greedy_losses_hn_D, 0)/(avg_slot_norm*self.model.num_slots)
        #     # obj_pd_greedy_loss_A = torch.cat([0.9*(x/(avg_slot_norm*self.model.num_slots)).mean(1)+0.1*y/(avg_slot_norm*self.model.num_slots) for x, y in zip(obj_pd_greedy_losses_en_A, obj_pd_greedy_losses_hn_A)], 0)
        #     # obj_pd_greedy_loss_D = torch.cat([0.9*(x/(avg_slot_norm*self.model.num_slots)).mean(1)+0.1*y/(avg_slot_norm*self.model.num_slots) for x, y in zip(obj_pd_greedy_losses_en_D, obj_pd_greedy_losses_hn_D)], 0)
        #     # avg_obj_pd_greedy_ratio_en = ((obj_pd_greedy_loss+obj_pd_greedy_loss_en_D).div(obj_pd_greedy_loss_en_D)).mean()
        #     # avg_obj_pd_greedy_ratio_hn = ((obj_pd_greedy_loss+obj_pd_greedy_loss_hn_D).div(obj_pd_greedy_loss_hn_D)).mean()
        #     # avg_obj_pd_greedy_ratio = ((obj_pd_greedy_loss+obj_pd_greedy_loss_D).div(obj_pd_greedy_loss_A)).mean()
        #     # std_obj_pd_greedy_loss = obj_pd_greedy_loss.std()/math.sqrt(obj_pd_greedy_loss.shape[0])
        #     # avg_obj_pd_greedy_loss = obj_pd_greedy_loss.mean()
        #     # avg_obj_pd_greedy_ctrast_en = avg_obj_pd_greedy_loss+obj_pd_greedy_loss_en_D.mean()
        #     # avg_obj_pd_greedy_ctrast_hn = avg_obj_pd_greedy_loss+obj_pd_greedy_loss_hn_D.mean()
        #     # avg_obj_pd_greedy_ctrast = avg_obj_pd_greedy_loss+obj_pd_greedy_loss_D.mean()

        #     # avg_attr_pd_greedy_loss = torch.cat(attr_pd_greedy_losses, 0)
        #     # avg_attr_pd_greedy_loss_en = (torch.cat(attr_pd_greedy_losses_en, 0)-avg_attr_pd_greedy_loss).mean()
        #     # avg_attr_pd_greedy_loss_hn = (torch.cat(attr_pd_greedy_losses_hn, 0)-avg_attr_pd_greedy_loss).mean()
        #     # std_attr_pd_greedy_loss = avg_attr_pd_greedy_loss.std()/math.sqrt(avg_attr_pd_greedy_loss.shape[0])
        #     # avg_attr_pd_greedy_loss = avg_attr_pd_greedy_loss.mean()

        #     # avg_obj_greedy_cos_loss = torch.cat(obj_greedy_cos_losses_nodup, 0).mean()
        #     # obj_greedy_cos_losses_nodup_en_A = torch.cat(obj_greedy_cos_losses_nodup_en_A, 0)
        #     # obj_greedy_cos_losses_nodup_en_D = torch.cat(obj_greedy_cos_losses_nodup_en_D, 0)
        #     # obj_greedy_cos_losses_nodup_hn_A = torch.cat(obj_greedy_cos_losses_nodup_hn_A, 0)
        #     # obj_greedy_cos_losses_nodup_hn_D = torch.cat(obj_greedy_cos_losses_nodup_hn_D, 0)
        #     # avg_obj_greedy_cos_loss_nodup_en = (avg_obj_greedy_cos_loss_nodup+torch.log(obj_greedy_cos_losses_nodup_en_D)-torch.log(obj_greedy_cos_losses_nodup_en_A)).mean()
        #     # avg_obj_greedy_cos_loss_nodup_hn = (avg_obj_greedy_cos_loss_nodup+torch.log(obj_greedy_cos_losses_nodup_hn_D)-torch.log(obj_greedy_cos_losses_nodup_hn_A)).mean()
        #     # avg_obj_greedy_cos_loss_nodup_norm = (avg_obj_greedy_cos_loss_nodup+torch.log(obj_greedy_cos_losses_nodup_hn_D+obj_greedy_cos_losses_nodup_en_D) \
        #     #                                     -torch.log(obj_greedy_cos_losses_nodup_hn_A+obj_greedy_cos_losses_nodup_en_A)).mean()
        #     # std_obj_greedy_cos_loss_nodup = avg_obj_greedy_cos_loss_nodup.std()/(math.sqrt(avg_obj_greedy_cos_loss_nodup.shape[0]))
        #     # avg_obj_greedy_cos_loss_nodup = avg_obj_greedy_cos_loss_nodup.mean()
        #     # avg_obj_greedy_cos_loss_norm_hn = avg_obj_greedy_cos_loss_nodup+torch.log(obj_greedy_cos_losses_nodup_hn_D).mean()

        #     # avg_attr_greedy_cos_loss_nodup = torch.cat(attr_greedy_cos_losses_nodup, 0)
        #     # avg_attr_greedy_cos_loss_nodup_en = (torch.cat(attr_greedy_cos_losses_nodup_en, 0)-avg_attr_greedy_cos_loss_nodup).mean()
        #     # avg_attr_greedy_cos_loss_nodup_hn = (torch.cat(attr_greedy_cos_losses_nodup_hn, 0)-avg_attr_greedy_cos_loss_nodup).mean()
        #     # std_attr_greedy_cos_loss_nodup = avg_attr_greedy_cos_loss_nodup.std()/(math.sqrt(avg_attr_greedy_cos_loss_nodup.shape[0]))
        #     # avg_attr_greedy_cos_loss_nodup = avg_attr_greedy_cos_loss_nodup.mean()

        #     # avg_obj_pd_greedy_cos_loss = torch.cat(obj_pd_greedy_cos_losses, 0)
        #     # avg_obj_pd_greedy_cos_loss_en = (torch.cat(obj_pd_greedy_cos_losses_en, 0)-avg_obj_pd_greedy_cos_loss).mean()
        #     # avg_obj_pd_greedy_cos_loss_hn = (torch.cat(obj_pd_greedy_cos_losses_hn, 0)-avg_obj_pd_greedy_cos_loss).mean()
        #     # std_obj_pd_greedy_cos_loss = avg_obj_pd_greedy_cos_loss.std()/(math.sqrt(avg_obj_pd_greedy_cos_loss.shape[0]))
        #     # avg_obj_pd_greedy_cos_loss = avg_obj_pd_greedy_cos_loss.mean()
        #     # avg_attr_pd_greedy_cos_loss = torch.cat(attr_pd_greedy_cos_losses, 0)
        #     # avg_attr_pd_greedy_cos_loss_en = (torch.cat(attr_pd_greedy_cos_losses_en, 0)-avg_attr_pd_greedy_cos_loss).mean()
        #     # avg_attr_pd_greedy_cos_loss_hn = (torch.cat(attr_pd_greedy_cos_losses_hn, 0)-avg_attr_pd_greedy_cos_loss).mean()
        #     # std_attr_pd_greedy_cos_loss = avg_attr_pd_greedy_cos_loss.std()/(math.sqrt(avg_attr_pd_greedy_cos_loss.shape[0]))
        #     # avg_attr_pd_greedy_cos_loss = avg_attr_pd_greedy_cos_loss.mean()

        #     logs = {
        #         # "avg_val_loss": avg_loss,
        #         "avg_ari_mask": avg_ari_mask,
        #         "avg_z_norm": avg_z_norm,
        #         "avg_z_angle": avg_z_angle,
        #         "avg_scaling": avg_scaling,
        #         "avg_angle": avg_angle,
        #         "avg_scaling_delta": avg_scaling_delta,
        #         "avg_angle_delta": avg_angle_delta,
        #         "avg_scaling_ratio": avg_scaling_ratio,
        #         "avg_angle_ratio": avg_angle_ratio,
        #         "avg_slot_std": avg_slot_std,
        #         "avg_obj_greedy_ctrast_en": avg_obj_greedy_ctrast_en,
        #         # "avg_obj_greedy_ctrast_hn": avg_obj_greedy_ctrast_hn,
        #         # "avg_obj_greedy_ctrast": avg_obj_greedy_ctrast,
        #         "avg_obj_greedy_loss": avg_obj_greedy_loss,
        #         # "avg_attr_greedy_loss_nodup": avg_attr_greedy_loss_nodup,
        #         # "avg_obj_pseudo_greedy_loss": avg_obj_pd_greedy_loss,
        #         # "avg_attr_pseudo_greedy_loss": avg_attr_pd_greedy_loss,
        #         "avg_obj_greedy_ratio_en": avg_obj_greedy_ratio_en,
        #         # "avg_attr_greedy_loss_nodup_en": avg_attr_greedy_loss_nodup_en,
        #         # "avg_obj_pseudo_greedy_loss_en": avg_obj_pd_greedy_loss_en,
        #         # "avg_attr_pseudo_greedy_loss_en": avg_attr_pd_greedy_loss_en,
        #         # "avg_obj_greedy_ratio_hn": avg_obj_greedy_ratio_hn,
        #         # "avg_attr_greedy_loss_nodup_hn": avg_attr_greedy_loss_nodup_hn,
        #         # "avg_obj_pseudo_greedy_loss_hn": avg_obj_pd_greedy_loss_hn,
        #         # "avg_attr_pseudo_greedy_loss_hn": avg_attr_pd_greedy_loss_hn,
        #         # "std_obj_pd_greedy_loss": std_obj_pd_greedy_loss,
        #         # "avg_obj_pd_greedy_ratio": avg_obj_pd_greedy_ratio,
        #         # "avg_obj_pd_greedy_ctrast_en": avg_obj_pd_greedy_ctrast_en,
        #         # "avg_obj_pd_greedy_ctrast_hn": avg_obj_pd_greedy_ctrast_hn,
        #         # "avg_obj_pd_greedy_ctrast": avg_obj_pd_greedy_ctrast,
        #         # "avg_obj_pd_greedy_loss": avg_obj_pd_greedy_loss,
        #         # "avg_obj_pd_greedy_ratio_en": avg_obj_pd_greedy_ratio_en,
        #         # "avg_obj_pd_greedy_ratio_hn": avg_obj_pd_greedy_ratio_hn,
        #         # "std_attr_greedy_loss_nodup": std_attr_greedy_loss_nodup,
        #         # "std_obj_pseudo_greedy_loss": std_obj_pd_greedy_loss,
        #         # "std_attr_pseudo_greedy_loss": std_attr_pd_greedy_loss,
        #         # "avg_obj_greedy_cos_loss_norm_hn": avg_obj_greedy_cos_loss_norm_hn,
        #         # "avg_obj_greedy_cos_loss": avg_obj_greedy_cos_loss,
        #         # "avg_attr_greedy_cos_loss_nodup": avg_attr_greedy_cos_loss_nodup,
        #         # "avg_obj_pseudo_greedy_cos_loss": avg_obj_pd_greedy_cos_loss,
        #         # "avg_attr_pseudo_greedy_cos_loss": avg_attr_pd_greedy_cos_loss,
        #         # "avg_obj_greedy_cos_loss_nodup_en": avg_obj_greedy_cos_loss_nodup_en,
        #         # "avg_attr_greedy_cos_loss_nodup_en": avg_attr_greedy_cos_loss_nodup_en,
        #         # "avg_obj_pseudo_greedy_cos_loss_en": avg_obj_pd_greedy_cos_loss_en,
        #         # "avg_attr_pseudo_greedy_cos_loss_en": avg_attr_pd_greedy_cos_loss_en,
        #         # "avg_obj_greedy_cos_loss_nodup_hn": avg_obj_greedy_cos_loss_nodup_hn,
        #         # "avg_attr_greedy_cos_loss_nodup_hn": avg_attr_greedy_cos_loss_nodup_hn,
        #         # "avg_obj_pseudo_greedy_cos_loss_hn": avg_obj_pd_greedy_cos_loss_hn,
        #         # "avg_attr_pseudo_greedy_cos_loss_hn": avg_attr_pd_greedy_cos_loss_hn,
        #         # "std_obj_greedy_cos_loss_nodup": std_obj_greedy_cos_loss_nodup,
        #         # "avg_obj_greedy_cos_loss_nodup_norm": avg_obj_greedy_cos_loss_nodup_norm,
        #         # "std_attr_greedy_cos_loss_nodup": std_attr_greedy_cos_loss_nodup,
        #         # "std_obj_pseudo_greedy_cos_loss": std_obj_pd_greedy_cos_loss,
        #         # "std_attr_pseudo_greedy_cos_loss": std_attr_pd_greedy_cos_loss,
        #     }
            logs = {"avg_ari_mask": avg_ari_mask,}
            # if self.trainer.running_sanity_check:
            #     self.trainer.running_sanity_check = False  # so that loggers don't skip logging
            #     self.log_dict(logs, sync_dist=True)
            #     self.trainer.current_epoch = 0
            # else:
            #     self.log_dict(logs, sync_dist=True)

            print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        warmup_steps = self.params.warmup_steps
        decay_steps = self.params.decay_steps
        total_steps = self.params.max_epochs * len(self.datamodule.train_dataloader())

        def warm_and_decay_lr_scheduler(step: int):
            # step = step * self.params.gpus # to make the decay consistent over multi-GPU
            # warmup_steps = self.params.warmup_steps * total_steps
            # decay_steps = self.params.decay_steps * total_steps
            # assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.params.scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )
