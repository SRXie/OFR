import pytorch_lightning as pl
import torch
import math
import numpy as np
from torch import optim
from torchvision import utils as vutils
from torchvision.transforms import transforms
from datetime import datetime


from beta_vae.model import BetaVAE
from beta_vae.params import BetaVAEParams
from beta_vae.utils import Tensor
from beta_vae.utils import split_and_interleave_stack
from beta_vae.utils import to_rgb_from_tensor, to_tensor_from_rgb
from beta_vae.utils import compute_loss, compute_cosine_loss, compute_partition_loss, compute_partition_cosine_loss


class BetaVAEMethod(pl.LightningModule):
    def __init__(self, model: BetaVAE, datamodule: pl.LightningDataModule, params: BetaVAEParams):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params
        self.save_hyperparameters('params')
        self.random_projection_init = False

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        results = self.forward(batch)
        train_loss = self.model.loss_function(*results,
                                            M_N = self.datamodule.train_batch_size/self.datamodule.num_train_images,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss

    def sample_images(self):
        dl = self.datamodule.obj_test_dataloader()
        random_idx = torch.randint(high=len(dl), size=(1,))
        batch = next(iter(dl))  # list of A, B, C, D, E -- E is the hard negative
        perm = torch.randperm(self.params.val_batch_size)
        idx = perm[: self.params.n_samples]
        batch = torch.cat([b[idx] for b in batch[:4]], 0)

        if self.params.gpus > 0:
            batch = batch.to(self.device)

        with torch.no_grad():

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

        return images

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        # batch is a list of lengthn num_slots+1
        with torch.no_grad():
            results = self.forward(batch[0])
            val_loss = self.model.loss_function(*results,
                                                M_N = self.datamodule.val_batch_size/self.datamodule.num_val_images,
                                                optimizer_idx = optimizer_idx,
                                                batch_idx = batch_idx)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # Algebra Test starts here
        odl = self.datamodule.obj_test_dataloader()
        adl = self.datamodule.attr_test_dataloader()

        z_norms = []
        obj_losses, obj_cos_losses, obj_acos_losses, attr_losses = [], [], [], []
        obj_losses_en_D, , obj_cos_losses_en_D,  obj_acos_losses_en_D, attr_losses_en = [], [], [], []

        def compute_test_losses(dataloader, losses, cos_losses, acos_losses, losses_en_D, cos_losses_en_D, acos_losses_D):
            b_prev = datetime.now()
            for batch in dataloader:
                # batch is a length-4 list, each element is a tensor of shape (batch_size, 3, width, height)
                batch_size = batch[0].shape[0]
                cat_batch = torch.cat(batch[:4], 0)
                if self.params.gpus > 0:
                    cat_batch = cat_batch.to(self.device)
                if self.trainer.running_sanity_check:
                    cat_zs = cat_batch.view(batch_size*2, -1)
                else:
                    mu, log_var = self.model.encode(cat_batch)
                    cat_zs = self.model.reparameterize(mu, log_var).detach()

                if dataloader is odl:
                    znorm = torch.norm(cat_zs, 2, -1)
                    znorm = torch.stack(torch.split(znorm, znorm.shape[0]//4, 0), 1).mean(1)
                    z_norms.append(znorm)

                cat_batch_EF = torch.cat(batch[4:], 0)
                if self.params.gpus > 0:
                    cat_batch_EF = cat_batch_EF.to(self.device)
                if self.trainer.running_sanity_check:
                    cat_zs_EF = cat_batch_EF.view(batch_size*2, -1)
                else:
                    EF_mu, EF_log_var = self.model.encode(cat_batch_EF)
                    cat_zs_EF = self.model.reparameterize(EF_mu, EF_log_var).detach()
                zs_E, zs_F = torch.split(cat_zs_EF, batch_size, 0)

                zs_A = cat_zs[:batch_size]
                compute_loss(cat_zs, losses)
                compute_cosine_loss(cat_zs, cos_losses, acos_losses)

                compute_partition_loss(cat_zs, losses_en_D)
                compute_partition_cosine_loss(cat_zs, obj_cos_en_D, obj_acos_en_D)
                # compute_partition_loss_hard(cat_zs, zs_E, zs_F, losses_hn_A, losses_hn_D)

        with torch.no_grad():
            compute_test_losses(odl, obj_losses, obj_cos_losses, obj_acos_losses, obj_losses_en_D, obj_cos_losses_en_D, obj_acos_losses_en_D)
            # compute_test_losses(adl, attr_losses, attr_losses_en, attr_losses_hn)

            avg_z_norm = torch.cat(z_norm, 0).mean()
            # avg_z_angle = torch.cat(z_angle, 0).mean()
            # avg_scaling = torch.cat(scalings, 0).mean()
            # avg_angle = torch.cat(angles, 0).mean()
            # avg_scaling_delta = torch.cat(scaling_deltas, 0).mean()
            # avg_angle_delta = torch.cat(angle_deltas, 0).mean()
            # avg_scaling_ratio = torch.cat(scaling_ratios, 0).mean()
            # avg_angle_ratio = torch.cat(angle_ratios, 0).mean()
            # slot_std = torch.cat(obj_greedy_std_nodup, 0)
            # avg_slot_std = slot_std.mean()

            obj_l2 = torch.cat(obj_losses, 0)
            obj_l2_en_D = torch.cat([x for x in obj_losses_en_D], 0)
            obj_l2_ratio = ((obj_l2_en_D-obj_l2).div(obj_l2_en_D))
            std_obj_l2_ratio = obj_l2_ratio.std()/math.sqrt(obj_l2_ratio.shape[0])
            avg_obj_l2_ratio = obj_l2_ratio.mean()
            avg_obj_l2 = obj_l2.mean()
            avg_obj_l2_ctrast_en = obj_l2_en_D.mean()-avg_obj_l2

            obj_cos = torch.cat(obj_cos_losses, 0)
            obj_cos_en_D = torch.cat([x for x in obj_cos_losses_en_D], 0)
            obj_cos_ratio = ((obj_cos_en_D-obj_cos_nodup).div(obj_cos_en_D))
            std_obj_cos_ratio = obj_cos_ratio.std()/math.sqrt(obj_cos_ratio.shape[0])
            avg_obj_cos_ratio = obj_cos_ratio.mean()
            avg_obj_cos = obj_cos.mean()
            avg_obj_cos_ctrast_en = obj_cos_nodup_en_D.mean()-avg_obj_cos

            obj_acos = torch.cat(obj_acos_losses, 0)
            obj_acos_en_D = torch.cat([x for x in obj_acos_losses_en_D], 0)
            obj_acos_ratio = ((obj_acos_en_D-obj_acos_nodup).div(obj_acos_en_D))
            std_obj_acos_ratio = obj_acos_ratio.std()/math.sqrt(obj_acos_ratio.shape[0])
            avg_obj_acos_ratio = obj_acos_ratio.mean()
            avg_obj_acos = obj_acos.mean()
            avg_obj_acos_ctrast_en = obj_acos_en_D.mean()-avg_obj_acos

            logs = {
                "avg_val_loss": avg_loss,
                "avg_ari_mask": avg_ari_mask,
                "avg_z_norm": avg_z_norm.to(self.device),
                "avg_z_angle": avg_z_angle.to(self.device),
                "avg_scaling": avg_scaling.to(self.device),
                "avg_angle": avg_angle.to(self.device),
                "avg_scaling_delta": avg_scaling_delta.to(self.device),
                "avg_angle_delta": avg_angle_delta.to(self.device),
                "avg_scaling_ratio": avg_scaling_ratio.to(self.device),
                "avg_angle_ratio": avg_angle_ratio.to(self.device),
                "avg_slot_std": avg_slot_std.to(self.device),
                "avg_obj_l2_ratio": avg_obj_l2_ratio.to(self.device),
                "avg_obj_l2": avg_obj_l2.to(self.device),
                "avg_obj_l2_ctrast_en": avg_obj_l2_ctrast_en.to(self.device),
                "std_obj_l2_ratio": std_obj_l2_ratio.to(self.device),
                "avg_obj_cos_ratio": avg_obj_cos_ratio.to(self.device),
                "avg_obj_cos": avg_obj_cos.to(self.device),
                "avg_obj_cos_ctrast_en": avg_obj_cos_ctrast_en.to(self.device),
                "std_obj_cos_ratio": std_obj_cos_ratio.to(self.device),
                "avg_obj_acos_ratio": avg_obj_acos_ratio.to(self.device),
                "avg_obj_acos": avg_obj_acos.to(self.device),
                "avg_obj_acos_ctrast_en": avg_obj_acos_ctrast_en.to(self.device),
                "std_obj_acos_ratio": std_obj_acos_ratio.to(self.device),
            }
            if self.trainer.running_sanity_check:
                self.trainer.running_sanity_check = False  # so that loggers don't skip logging
                self.log_dict(logs, sync_dist=True)
                self.trainer.current_epoch = 0
            else:
                self.log_dict(logs, sync_dist=True)

            print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.params.scheduler_gamma)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "epoch",}],
        )
