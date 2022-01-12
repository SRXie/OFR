import pytorch_lightning as pl
import torch
import math
import numpy as np
from torch import optim
from torchvision import utils as vutils
from torchvision.transforms import transforms

from params import Params
from utils import Tensor
from utils import to_rgb_from_tensor, to_tensor_from_rgb
from utils import compute_cos_distance, compute_rank_correlation
from utils import batched_index_select
from utils import compute_greedy_loss, compute_loss, compute_cosine_loss
from utils import compute_shuffle_greedy_loss, compute_shuffle_loss, compute_shuffle_cosine_loss
from utils import compute_all_losses, summarize_losses
from utils import swap_bg_slot_back
from utils import captioned_masked_recons
from utils import split_and_interleave_stack
from utils import compute_corr_coef


class ObjTestMethod(pl.LightningModule):
    def __init__(self, model, datamodule: pl.LightningDataModule, params: Params):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params
        self.save_hyperparameters('params')
        self.random_projection_init = False

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if self.params.model == "btc-vae":
            results = self.forward(batch)
            train_loss = self.model.loss_function(*results,
                                            M_N = self.datamodule.train_batch_size/self.datamodule.num_train_images,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
            logs = {key: val.item() for key, val in train_loss.items()}
            self.log_dict(logs, sync_dist=True)
        elif self.params.model == "slot-attn" or "iodine" in self.params.model:
            train_loss = self.model.loss_function(batch)
            logs = {key: val.item() for key, val in train_loss.items()}
            self.log_dict(logs, sync_dist=True)
        else:
            raise NotImplementedError
        return train_loss

    def sample_images(self):
        dl = self.datamodule.obj_test_dataloader()
        random_idx = torch.randint(high=len(dl), size=(1,))
        batch = next(iter(dl))  # list of A, B, C, D, E, F -- E and F are hard negatives
        perm = torch.randperm(self.params.test_batch_size)
        idx = perm[: self.params.n_samples]
        batch = torch.cat([b[idx] for b in batch[:4]], 0)

        if self.params.gpus > 0:
            batch = batch.to(self.device)

        if True:

            if self.params.model == "btc-vae":
                recons = self.model.generate(batch)
                batch = split_and_interleave_stack(batch, self.params.n_samples)
                recons = split_and_interleave_stack(recons, self.params.n_samples)
                # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
                out = to_rgb_from_tensor(
                    torch.cat(
                        [
                            batch.unsqueeze(1),  # original images
                            recons.unsqueeze(1),  # reconstructions
                        ],
                        dim=1,
                    )
                )

                batch_size, C, H, W = recons.shape
                images = vutils.make_grid(
                    out.view(batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
                )

            else:
                if self.params.model == "slot-attn":
                    recon_combined, recons, masks, slots, attns, recon_combined_nodup, recons_nodup, masks_nodup, slots_nodup = self.model.forward(batch, dup_threshold=self.params.dup_threshold, viz=True)
                elif "iodine" in self.params.model:
                    recon_combined, recons, masks, slots, recon_combined_nodup, recons_nodup, masks_nodup, slots_nodup = self.model.forward(batch, dup_threshold=self.params.dup_threshold, viz=True)
                else:
                    raise NotImplementedError

                # throw background slot back
                if self.params.model == "slot-attn":
                    cat_indices = swap_bg_slot_back(attns)
                    recons = batched_index_select(recons, 1, cat_indices)
                    masks = batched_index_select(masks, 1, cat_indices)
                    slots = batched_index_select(slots, 1, cat_indices)
                    attns = batched_index_select(attns, 2, cat_indices)
                # recons_nodup = batched_index_select(recons_nodup, 1, cat_indices)
                # masks_nodup = batched_index_select(masks_nodup, 1, cat_indices)
                # slots_nodup = batched_index_select(slots_nodup, 1, cat_indices)

                # reorder with matching
                _, cat_indices = compute_greedy_loss(slots)
                recons_perm = batched_index_select(recons, 1, cat_indices)
                masks_perm = batched_index_select(masks, 1, cat_indices)
                slots_perm = batched_index_select(slots, 1, cat_indices)
                if self.params.model == "slot-attn":
                    attns_perm = batched_index_select(attns, 2, cat_indices)
                    masked_recons_perm, masked_attn_perm, recons_perm = captioned_masked_recons(recons_perm, masks_perm, slots_perm, attns_perm)
                elif "iodine" in self.params.model:
                    masked_recons_perm, masked_attn_perm, recons_perm = captioned_masked_recons(recons_perm, masks_perm, slots_perm)
                else:
                    raise NotImplementedError
                # No need to match again
                # cat_indices_nodup = compute_greedy_loss(slots_nodup, [])
                batch_list = []
                batch_nodup = torch.cat((batch, batch[-self.params.n_samples:]), 0)
                recons_perm_nodup = recons_nodup #batched_index_select(recons_nodup, 1, cat_indices_nodup)
                masks_perm_nodup = masks_nodup #batched_index_select(masks_nodup, 1, cat_indices_nodup)
                slots_perm_nodup = slots_nodup #batched_index_select(slots_nodup, 1, cat_indices_nodup)
                if self.params.model == "slot-attn":
                    attns_perm_nodup = torch.cat((attns, attns[-self.params.n_samples:]), 0)  #batched_index_select(attns, 2, cat_indices_nodup)
                    masked_recons_perm_nodup, masked_attn_perm_nodup, recons_perm_nodup = captioned_masked_recons(recons_perm_nodup, masks_perm_nodup, slots_perm_nodup, attns_perm_nodup)
                elif "iodine" in self.params.model:
                    masked_recons_perm_nodup, masked_attn_perm_nodup, recons_perm_nodup = captioned_masked_recons(recons_perm_nodup, masks_perm_nodup, slots_perm_nodup)
                else:
                    raise NotImplementedError
                batch = split_and_interleave_stack(batch, self.params.n_samples)
                recon_combined = split_and_interleave_stack(recon_combined, self.params.n_samples)
                recons_perm = split_and_interleave_stack(recons_perm, self.params.n_samples)
                masks_perm = split_and_interleave_stack(masks_perm, self.params.n_samples)
                slots_perm = split_and_interleave_stack(slots_perm, self.params.n_samples)
                masked_attn_perm = split_and_interleave_stack(masked_attn_perm, self.params.n_samples)
                batch_nodup = split_and_interleave_stack(batch_nodup, self.params.n_samples)
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
                            torch.cat([batch.unsqueeze(1), batch_nodup.unsqueeze(1)], dim=0),  # original images
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
                    out.view((2 * batch_size +batch_size//4) * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
                )

            return images

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        # batch is a list of lengthn num_slots+1
        if True:
            if self.params.model == "btc-vae":
                results = self.forward(batch[0])
                val_loss = self.model.loss_function(*results,
                                                M_N = self.datamodule.val_batch_size/self.datamodule.num_val_images,
                                                optimizer_idx = optimizer_idx,
                                                batch_idx = batch_idx)
            elif self.params.model == "slot-attn" or "iodine" in self.params.model:
                val_loss = self.model.loss_function(batch[0], batch[1:-1], batch[-1])
            else:
                raise NotImplementedError
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if self.params.model == "slot-attn" or "iodine" in self.params.model:
            avg_ari_mask = np.stack([x["mask_ari"] for x in outputs]).mean()

        # Algebra Test starts here
        dl = self.datamodule.obj_test_dataloader()

        obj_losses, obj_losses_en_D = [], []
        obj_cos_losses, obj_cos_losses_en_D = [], []
        obj_acos_losses, obj_acos_losses_en_D = [], []

        obj_losses_hn, color_losses_hn, mat_losses_hn, shape_losses_hn, size_losses_hn, snd_losses_hn = [], [], [], [], [], []
        hn_losses_list = [obj_losses_hn, color_losses_hn, mat_losses_hn, shape_losses_hn, size_losses_hn, snd_losses_hn]
        obj_cos_losses_hn, color_cos_losses_hn, mat_cos_losses_hn, shape_cos_losses_hn, size_cos_losses_hn, snd_cos_losses_hn = [], [], [], [], [], []
        hn_cos_losses_list = [obj_cos_losses_hn, color_cos_losses_hn, mat_cos_losses_hn, shape_cos_losses_hn, size_cos_losses_hn, snd_cos_losses_hn]
        obj_acos_losses_hn, color_acos_losses_hn, mat_acos_losses_hn, shape_acos_losses_hn, size_acos_losses_hn, snd_acos_losses_hn = [], [], [], [], [], []
        hn_acos_losses_list = [obj_acos_losses_hn, color_acos_losses_hn, mat_acos_losses_hn, shape_acos_losses_hn, size_acos_losses_hn, snd_acos_losses_hn]

        if True:
            if self.params.model == "btc-vae":
                for batch in dl:
                    # batch is a length-4 list, each element is a tensor of shape (batch_size, 3, width, height)
                    batch_size = batch[0].shape[0]
                    cat_batch = torch.cat(batch, 0)
                    if self.params.gpus > 0:
                        cat_batch = cat_batch.to(self.device)

                    mu, log_var = self.model.encode(cat_batch)
                    cat_zs = mu #self.model.reparameterize(mu, log_var).detach()

                    # if self.trainer.running_sanity_check:
                    #     compute_loss(cat_batch[:4*batch_size].view(4*batch_size, -1), obj_losses)
                    #     compute_cosine_loss(cat_batch[:4*batch_size].view(4*batch_size, -1), obj_cos_losses, obj_acos_losses)

                    #     compute_shuffle_loss(cat_batch[:4*batch_size].view(4*batch_size, -1), obj_losses_en_D)
                    #     compute_shuffle_cosine_loss(cat_batch[:4*batch_size].view(4*batch_size, -1), obj_cos_losses_en_D, obj_acos_losses_en_D)

                    # else:
                    compute_loss(cat_zs[:4*batch_size], obj_losses)
                    compute_cosine_loss(cat_zs[:4*batch_size], obj_cos_losses, obj_acos_losses)

                    compute_shuffle_loss(cat_zs[:4*batch_size], obj_losses_en_D)
                    compute_shuffle_cosine_loss(cat_zs[:4*batch_size], obj_cos_losses_en_D, obj_acos_losses_en_D)

                    for ind in range(4, 9):
                        zs_D_prime = cat_zs[ind*batch_size:(ind+1)*batch_size]
                        cat_zs_hn = torch.cat((cat_zs[:3*batch_size], zs_D_prime), 0)
                        compute_loss(cat_zs_hn, hn_losses_list[ind-4])
                        compute_cosine_loss(cat_zs_hn, hn_cos_losses_list[ind-4], hn_acos_losses_list[ind-4])
            else:
                slots_nodup_list = []
                dup_threshold = self.params.dup_threshold
                i = 0
                for batch in dl:
                    # batch is a length-9 list, each element is a tensor of shape (batch_size, 3, width, height)
                    i += 0.01
                    batch_size = batch[0].shape[0]
                    cat_batch = torch.cat(batch+[batch[3]], 0)
                    if self.params.gpus > 0:
                        cat_batch = cat_batch.to(self.device)

                    if self.params.model == "slot-attn":
                        _, _, _, slots, _, _, _, _, slots_nodup = self.model.forward(cat_batch, slots_only=False, dup_threshold=dup_threshold, viz=False)
                    elif "iodine" in self.params.model:
                        _, _, _, slots, _, _, _, slots_nodup = self.model.forward(cat_batch, dup_threshold=dup_threshold, viz=False)
                    else:
                        raise NotImplementedError

                    slots_nodup = slots_nodup.cpu()
                    slots_nodup_list.append(slots_nodup)
                    if len(slots_nodup_list)==64//self.params.test_batch_size:
                        cat_slots_full = torch.cat(slots_nodup_list, 0)
                        cat_slots_full = split_and_interleave_stack(cat_slots_full, slots_nodup.shape[0])
                        slots_nodup_list = []
                    else:
                        continue
                    # Here we compute the angle between DC and D'C
                    batch_size*=16

                    cat_slots = cat_slots_full[:4*batch_size]
                    losses, cos_losses, acos_losses, cat_slots = compute_all_losses(cat_slots)
                    obj_losses.append(losses)
                    obj_cos_losses.append(cos_losses)
                    obj_acos_losses.append(acos_losses)

                    losses_en_D, cos_losses_en_D, acos_losses_en_D = compute_shuffle_greedy_loss(cat_slots)
                    obj_losses_en_D.append(losses_en_D)
                    obj_cos_losses_en_D.append(cos_losses_en_D)
                    obj_acos_losses_en_D.append(acos_losses_en_D)

                    for ind in range(4, 9):
                        slots_D_prime = at_slots_full[ind*batch_size:(ind+1)*batch_size]
                        cat_slots = torch.cat((cat_slots[:3*batch_size], slots_D_prime), 0)
                        losses, cos_losses, acos_losses, _ = compute_all_losses(cat_slots)
                        hn_losses_list[ind-4].append(losses)
                        hn_cos_losses_list[ind-4].append(cos_losses)
                        hn_acos_losses_list[ind-4].append(acos_losses)

            std_obj_l2_ratio, avg_obj_l2_ratio, avg_obj_l2, avg_obj_l2_baseline = summarize_losses(obj_losses, obj_losses_en_D)
            std_obj_cos_ratio, avg_obj_cos_ratio, avg_obj_cos, avg_obj_cos_baseline = summarize_losses(obj_cos_losses, obj_cos_losses_en_D)
            std_obj_acos_ratio, avg_obj_acos_ratio, avg_obj_acos, avg_obj_acos_baseline = summarize_losses(obj_acos_losses, obj_acos_losses_en_D)

            _, avg_obj_l2_hn, _, _ = summarize_losses(obj_losses_hn, obj_losses_en_D)
            _, avg_obj_cos_hn, _, _ = summarize_losses(obj_cos_losses_hn, obj_cos_losses_en_D)
            _, avg_obj_acos_hn, _, _ = summarize_losses(obj_acos_losses_hn, obj_acos_losses_en_D)

            _, avg_color_l2_hn, _, _ = summarize_losses(color_losses_hn, obj_losses_en_D)
            _, avg_color_cos_hn, _, _ = summarize_losses(color_cos_losses_hn, obj_cos_losses_en_D)
            _, avg_color_acos_hn, _, _ = summarize_losses(color_acos_losses_hn, obj_acos_losses_en_D)

            _, avg_mat_l2_hn, _, _ = summarize_losses(mat_losses_hn, obj_losses_en_D)
            _, avg_mat_cos_hn, _, _ = summarize_losses(mat_cos_losses_hn, obj_cos_losses_en_D)
            _, avg_mat_acos_hn, _, _ = summarize_losses(mat_acos_losses_hn, obj_acos_losses_en_D)

            _, avg_shape_l2_hn, _, _ = summarize_losses(shape_losses_hn, obj_losses_en_D)
            _, avg_shape_cos_hn, _, _ = summarize_losses(shape_cos_losses_hn, obj_cos_losses_en_D)
            _, avg_shape_acos_hn, _, _ = summarize_losses(shape_acos_losses_hn, obj_acos_losses_en_D)

            _, avg_size_l2_hn, _, _ = summarize_losses(size_losses_hn, obj_losses_en_D)
            _, avg_size_cos_hn, _, _ = summarize_losses(size_cos_losses_hn, obj_cos_losses_en_D)
            _, avg_size_acos_hn, _, _ = summarize_losses(size_acos_losses_hn, obj_acos_losses_en_D)

            logs = {
                "avg_val_loss": avg_loss,
                "avg_obj_l2_ratio": avg_obj_l2_ratio.to(self.device),
                "avg_obj_l2": avg_obj_l2.to(self.device),
                "avg_obj_l2_baseline": avg_obj_l2_baseline.to(self.device),
                "std_obj_l2_ratio": std_obj_l2_ratio.to(self.device),
                "avg_obj_l2_baseline_hn": (avg_obj_l2_hn).to(self.device),
                "avg_color_l2_baseline_hn": (avg_color_l2_hn).to(self.device),
                "avg_mat_l2_baseline_hn": (avg_mat_l2_hn).to(self.device),
                "avg_shape_l2_baseline_hn": (avg_shape_l2_hn).to(self.device),
                "avg_size_l2_baseline_hn": (avg_size_l2_hn).to(self.device),
                "avg_obj_l2_ratio_hn": (1-avg_obj_l2/avg_obj_l2_hn).to(self.device),
                "avg_color_l2_ratio_hn": (1-avg_obj_l2/avg_color_l2_hn).to(self.device),
                "avg_mat_l2_ratio_hn": (1-avg_obj_l2/avg_mat_l2_hn).to(self.device),
                "avg_shape_l2_ratio_hn": (1-avg_obj_l2/avg_shape_l2_hn).to(self.device),
                "avg_size_l2_ratio_hn": (1-avg_obj_l2/avg_size_l2_hn).to(self.device),
                "avg_obj_cos_ratio": avg_obj_cos_ratio.to(self.device),
                "avg_obj_cos": avg_obj_cos.to(self.device),
                "avg_obj_cos_baseline": avg_obj_cos_baseline.to(self.device),
                "std_obj_cos_ratio": std_obj_cos_ratio.to(self.device),
                "avg_obj_cos_baseline_hn": (avg_obj_cos_hn).to(self.device),
                "avg_color_cos_baseline_hn": (avg_color_cos_hn).to(self.device),
                "avg_mat_cos_baseline_hn": (avg_mat_cos_hn).to(self.device),
                "avg_shape_cos_baseline_hn": (avg_shape_cos_hn).to(self.device),
                "avg_size_cos_baseline_hn": (avg_size_cos_hn).to(self.device),
                "avg_obj_cos_ratio_hn": (1-avg_obj_cos/avg_obj_cos_hn).mean().to(self.device),
                "avg_color_cos_ratio_hn": (1-avg_obj_cos/avg_color_cos_hn).to(self.device),
                "avg_mat_cos_ratio_hn": (1-avg_obj_cos/avg_mat_cos_hn).to(self.device),
                "avg_shape_cos_ratio_hn": (1-avg_obj_cos/avg_shape_cos_hn).to(self.device),
                "avg_size_cos_ratio_hn": (1-avg_obj_cos/avg_size_cos_hn).to(self.device),,
                "avg_obj_acos_ratio": avg_obj_acos_ratio.to(self.device),
                "avg_obj_acos": avg_obj_acos.to(self.device),
                "avg_obj_acos_baseline": avg_obj_acos_baseline.to(self.device),
                "std_obj_acos_ratio": std_obj_acos_ratio.to(self.device),
                "avg_obj_acos_baseline_hn": (avg_obj_acos_hn).mean().to(self.device),
                "avg_color_acos_baseline_hn": (avg_color_acos_hn).to(self.device),
                "avg_mat_acos_baseline_hn": (avg_mat_acos_hn).to(self.device),
                "avg_shape_acos_baseline_hn": (avg_shape_acos_hn).to(self.device),
                "avg_size_acos_baseline_hn": (avg_size_acos_hn).to(self.device),
                "avg_obj_acos_ratio_hn": (1-avg_obj_acos/avg_obj_acos_hn).to(self.device),
                "avg_color_acos_ratio_hn": (1-avg_obj_acos/avg_color_acos_hn).to(self.device),
                "avg_mat_acos_ratio_hn": (1-avg_obj_acos/avg_mat_acos_hn).to(self.device),
                "avg_shape_acos_ratio_hn": (1-avg_obj_acos/avg_shape_acos_hn).to(self.device),
                "avg_size_acos_ratio_hn": (1-avg_obj_acos/avg_size_acos_hn).to(self.device),
            }
            if self.params.model == "slot-attn" or "iodine" in self.params.model:
                logs["avg_ari_mask"] = avg_ari_mask
            if self.trainer.running_sanity_check:
                self.trainer.running_sanity_check = False  # so that loggers don't skip logging
                self.log_dict(logs, sync_dist=True)
                self.trainer.current_epoch = 0
            else:
                self.log_dict(logs, sync_dist=True)

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
