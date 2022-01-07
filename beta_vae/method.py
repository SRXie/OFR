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
from beta_vae.utils import compute_loss, compute_cosine_loss, compute_shuffle_loss, compute_shuffle_cosine_loss
from slot_attention.utils import summarize_losses

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
        dl = self.datamodule.obj_test_dataloader()

        obj_losses, obj_cos_losses, obj_acos_losses = [], [], []
        obj_losses_en_D, obj_cos_losses_en_D,  obj_acos_losses_en_D, attr_losses_en = [], [], [], []

        obj_losses_hn, color_losses_hn, mat_losses_hn, shape_losses_hn, size_losses_hn = [], [], [], [], []
        hn_losses_list = [obj_losses_hn, color_losses_hn, mat_losses_hn, shape_losses_hn, size_losses_hn]
        obj_cos_losses_hn, color_cos_losses_hn, mat_cos_losses_hn, shape_cos_losses_hn, size_cos_losses_hn = [], [], [], [], []
        hn_cos_losses_list = [obj_cos_losses_hn, color_cos_losses_hn, mat_cos_losses_hn, shape_cos_losses_hn, size_cos_losses_hn]
        obj_acos_losses_hn, color_acos_losses_hn, mat_acos_losses_hn, shape_acos_losses_hn, size_acos_losses_hn = [], [], [], [], []
        hn_acos_losses_list = [obj_acos_losses_hn, color_acos_losses_hn, mat_acos_losses_hn, shape_acos_losses_hn, size_acos_losses_hn]

        with torch.no_grad():
            for batch in dl:
                # batch is a length-4 list, each element is a tensor of shape (batch_size, 3, width, height)
                batch_size = batch[0].shape[0]
                cat_batch = torch.cat(batch, 0)
                if self.params.gpus > 0:
                    cat_batch = cat_batch.to(self.device)

                mu, log_var = self.model.encode(cat_batch)
                cat_zs = mu #self.model.reparameterize(mu, log_var).detach()

                compute_loss(cat_zs[:4*batch_size], obj_losses)
                compute_cosine_loss(cat_zs[:4*batch_size], obj_cos_losses, obj_acos_losses)

                compute_shuffle_loss(cat_zs[:4*batch_size], obj_losses_en_D)
                compute_shuffle_cosine_loss(cat_zs[:4*batch_size], obj_cos_losses_en_D, obj_acos_losses_en_D)

                for ind in range(4, 9):
                    if ind == 4:
                        zs_D_prime = cat_zs[-batch_size:]
                    else:
                        zs_D_prime = cat_zs[3*batch_size:4*batch_size] #cat_zs[ind*batch_size:(ind+1)*batch_size]
                    cat_zs_hn = torch.cat((cat_zs[:3*batch_size], zs_D_prime), 0)
                    compute_loss(cat_zs_hn, hn_losses_list[ind-4])
                    compute_cosine_loss(cat_zs_hn, hn_cos_losses_list[ind-4], hn_acos_losses_list[ind-4])

            std_obj_l2_ratio, avg_obj_l2_ratio, avg_obj_l2, avg_obj_l2_ctrast_en = summarize_losses(obj_losses, obj_losses_en_D)
            std_obj_cos_ratio, avg_obj_cos_ratio, avg_obj_cos, avg_obj_cos_ctrast_en = summarize_losses(obj_cos_losses, obj_cos_losses_en_D)
            std_obj_acos_ratio, avg_obj_acos_ratio, avg_obj_acos, avg_obj_acos_ctrast_en = summarize_losses(obj_acos_losses, obj_acos_losses_en_D)

            _, avg_obj_l2_hn_ratio, _, _ = summarize_losses(obj_losses_hn, obj_losses_en_D)
            _, avg_obj_cos_hn_ratio, _, _ = summarize_losses(obj_cos_losses_hn, obj_cos_losses_en_D)
            _, avg_obj_acos_hn_ratio, _, _ = summarize_losses(obj_acos_losses_hn, obj_acos_losses_en_D)

            _, avg_color_l2_hn_ratio, _, _ = summarize_losses(color_losses_hn, obj_losses_en_D)
            _, avg_color_cos_hn_ratio, _, _ = summarize_losses(color_cos_losses_hn, obj_cos_losses_en_D)
            _, avg_color_acos_hn_ratio, _, _ = summarize_losses(color_acos_losses_hn, obj_acos_losses_en_D)

            _, avg_mat_l2_hn_ratio, _, _ = summarize_losses(mat_losses_hn, obj_losses_en_D)
            _, avg_mat_cos_hn_ratio, _, _ = summarize_losses(mat_cos_losses_hn, obj_cos_losses_en_D)
            _, avg_mat_acos_hn_ratio, _, _ = summarize_losses(mat_acos_losses_hn, obj_acos_losses_en_D)

            _, avg_shape_l2_hn_ratio, _, _ = summarize_losses(shape_losses_hn, obj_losses_en_D)
            _, avg_shape_cos_hn_ratio, _, _ = summarize_losses(shape_cos_losses_hn, obj_cos_losses_en_D)
            _, avg_shape_acos_hn_ratio, _, _ = summarize_losses(shape_acos_losses_hn, obj_acos_losses_en_D)

            _, avg_size_l2_hn_ratio, _, _ = summarize_losses(size_losses_hn, obj_losses_en_D)
            _, avg_size_cos_hn_ratio, _, _ = summarize_losses(size_cos_losses_hn, obj_cos_losses_en_D)
            _, avg_size_acos_hn_ratio, _, _ = summarize_losses(size_acos_losses_hn, obj_acos_losses_en_D)

            logs = {
                "avg_val_loss": avg_loss,
                "avg_obj_l2_ratio": avg_obj_l2_ratio.to(self.device),
                "avg_obj_l2": avg_obj_l2.to(self.device),
                "avg_obj_l2_ctrast_en": avg_obj_l2_ctrast_en.to(self.device),
                "std_obj_l2_ratio": std_obj_l2_ratio.to(self.device),
                "avg_obj_l2_gap": (avg_obj_l2_hn_ratio-avg_obj_l2_ratio).to(self.device),
                "avg_color_l2_gap": (avg_color_l2_hn_ratio-avg_obj_l2_ratio).to(self.device),
                "avg_mat_l2_gap": (avg_mat_l2_hn_ratio-avg_obj_l2_ratio).to(self.device),
                "avg_shape_l2_gap": (avg_shape_l2_hn_ratio-avg_obj_l2_ratio).to(self.device),
                "avg_size_l2_gap": (avg_size_l2_hn_ratio-avg_obj_l2_ratio).to(self.device),
                "avg_obj_cos_ratio": avg_obj_cos_ratio.to(self.device),
                "avg_obj_cos": avg_obj_cos.to(self.device),
                "avg_obj_cos_ctrast_en": avg_obj_cos_ctrast_en.to(self.device),
                "std_obj_cos_ratio": std_obj_cos_ratio.to(self.device),
                "avg_obj_cos_gap": (avg_obj_cos_hn_ratio-avg_obj_cos_ratio).to(self.device),
                "avg_color_cos_gap": (avg_color_cos_hn_ratio-avg_obj_cos_ratio).to(self.device),
                "avg_mat_cos_gap": (avg_mat_cos_hn_ratio-avg_obj_cos_ratio).to(self.device),
                "avg_shape_cos_gap": (avg_shape_cos_hn_ratio-avg_obj_cos_ratio).to(self.device),
                "avg_size_cos_gap": (avg_size_cos_hn_ratio-avg_obj_cos_ratio).to(self.device),
                "avg_obj_acos_ratio": avg_obj_acos_ratio.to(self.device),
                "avg_obj_acos": avg_obj_acos.to(self.device),
                "avg_obj_acos_ctrast_en": avg_obj_acos_ctrast_en.to(self.device),
                "std_obj_acos_ratio": std_obj_acos_ratio.to(self.device),
                "avg_obj_acos_gap": (avg_obj_acos_hn_ratio-avg_obj_acos_ratio).to(self.device),
                "avg_color_acos_gap": (avg_color_acos_hn_ratio-avg_obj_acos_ratio).to(self.device),
                "avg_mat_acos_gap": (avg_mat_acos_hn_ratio-avg_obj_acos_ratio).to(self.device),
                "avg_shape_acos_gap": (avg_shape_acos_hn_ratio-avg_obj_acos_ratio).to(self.device),
                "avg_size_acos_gap": (avg_size_acos_hn_ratio-avg_obj_acos_ratio).to(self.device),
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
