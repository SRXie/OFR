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
from beta_vae.utils import compute_loss, compute_partition_loss, compute_partition_loss_hard


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
        obj_losses, attr_losses = [], []
        obj_losses_en_A, obj_losses_en_D, attr_losses_en = [], [], []
        obj_losses_hn_A, obj_losses_hn_D, attr_losses_hn = [], [], []

        def compute_test_losses(dataloader, losses, losses_en_A, losses_en_D, losses_hn_A, losses_hn_D):
            b_prev = datetime.now()
            for batch in dataloader:
                print("load data:", datetime.now()-b_prev)
                # sample_losses = []
                # batch is a length-4 list, each element is a tensor of shape (batch_size, 3, width, height)
                batch_size = batch[0].shape[0]
                cat_batch = torch.cat(batch[:4], 0)
                if self.params.gpus > 0:
                    cat_batch = cat_batch.to(self.device)
                cat_zs = self.model.encode(cat_batch)[0]

                if dataloader is odl:
                    znorm = torch.norm(cat_zs, 2, -1)
                    znorm = torch.stack(torch.split(znorm, znorm.shape[0]//4, 0), 1).mean(1)
                    z_norms.append(znorm)

                cat_batch_EF = torch.cat(batch[4:], 0)
                if self.params.gpus > 0:
                    cat_batch_EF = cat_batch_EF.to(self.device)
                cat_zs_EF = self.model.encode(cat_batch_EF)[0]
                zs_E, zs_F = torch.split(cat_zs_EF, batch_size, 0)

                compute_loss(cat_zs, losses)
                compute_partition_loss(cat_zs, losses_en_A, losses_en_D)
                compute_partition_loss_hard(cat_zs, zs_E, zs_F, losses_hn_A, losses_hn_D)

                print("batch time:", datetime.now()-b_prev)
                b_prev = datetime.now()

        with torch.no_grad():
            compute_test_losses(odl, obj_losses, obj_losses_en_A, obj_losses_en_D, obj_losses_hn_A, obj_losses_hn_D)
            # compute_test_losses(adl, attr_losses, attr_losses_en, attr_losses_hn)

            avg_z_norm = torch.cat(z_norms).mean()
            obj_loss = torch.cat(obj_losses, 0)/avg_z_norm
            obj_loss_en_A = torch.cat([torch.log(torch.exp(x/avg_z_norm).sum(1)) for x in obj_losses_en_A], 0)
            obj_loss_en_D = torch.cat([torch.log(torch.exp(x/avg_z_norm).sum(1)) for x in obj_losses_en_D], 0)
            obj_loss_hn_A = torch.cat([torch.log(torch.exp(x/avg_z_norm)) for x in obj_losses_hn_A], 0)
            obj_loss_hn_D = torch.cat([torch.log(torch.exp(x/avg_z_norm)) for x in obj_losses_hn_D], 0)
            avg_obj_loss_en = (obj_loss+obj_loss_en_D-obj_loss_en_A).mean()
            avg_obj_loss_hn = (obj_loss+obj_loss_hn_D-obj_loss_hn_A).mean()
            avg_obj_loss_norm = (obj_loss+obj_loss_en_D+obj_loss_hn_D-obj_loss_en_A-obj_loss_hn_A).mean()
            std_obj_loss = obj_loss.std()/math.sqrt(obj_loss.shape[0])
            avg_obj_loss = obj_loss.mean()
            avg_obj_loss_nll_en = avg_obj_loss+obj_loss_en_D.mean()
            avg_obj_loss_nll_hn = avg_obj_loss+obj_loss_hn_D.mean()

            # avg_attr_loss = torch.cat(attr_losses, 0)
            # avg_attr_loss_en = (torch.cat(attr_losses_en, 0)-avg_attr_loss).mean()
            # avg_attr_loss_hn = (torch.cat(attr_losses_hn, 0)-avg_attr_loss).mean()
            # std_attr_loss = avg_attr_loss.std()/math.sqrt(avg_attr_loss.shape[0])
            # avg_attr_loss = avg_attr_loss.mean()

            logs = {
                "avg_z_norms": avg_z_norm,
                "avg_val_loss": avg_loss,
                "avg_obj_loss": avg_obj_loss,
                # "avg_attr_loss": avg_attr_loss,
                "avg_obj_loss_norm": avg_obj_loss_norm,
                "avg_obj_loss_nll_en": avg_obj_loss_nll_en,
                "avg_obj_loss_nll_hn":avg_obj_loss_nll_hn,
                "avg_obj_loss_en": avg_obj_loss_en,
                # "avg_attr_loss_en": avg_attr_loss_en,
                "avg_obj_loss_hn": avg_obj_loss_hn,
                # "avg_attr_loss_hn": avg_attr_loss_hn,
                "std_obj_loss": std_obj_loss,
                # "std_attr_loss": std_attr_loss,
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
