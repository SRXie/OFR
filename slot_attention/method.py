import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils

from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import Tensor
from slot_attention.utils import to_rgb_from_tensor


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
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.batch_size)
        idx = perm[: self.params.n_samples]
        batch = next(iter(dl))[idx]
        if self.params.gpus > 0:
            batch = batch.to(self.device)
        recon_combined, recons, masks, slots = self.model.forward(batch)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            )
        )

        batch_size, num_slots, C, H, W = recons.shape
        images = vutils.make_grid(
            out.view(batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
        )

        return images

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        val_loss = self.model.loss_function(batch)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # Algebra Test starts here
        odl = self.datamodule.obj_test_dataloader()
        adl = self.datamodule.attr_test_dataloader()
        sample_size = 10000
        obj_rand_aggr_losses_epoch, obj_greedy_losses_epoch, obj_sample_losses_epoch = [], [], []
        attr_rand_aggr_losses_epoch, attr_greedy_losses_epoch, attr_sample_losses_epoch = [], [], []

        rand = torch.rand(self.params.batch_size*sample_size*3, self.params.num_slots)
        batch_rand_perm = rand.argsort(dim=1)
        del rand
        if self.params.gpus > 0:
            batch_rand_perm = batch_rand_perm.to(self.device)

        def compute_test_losses(batch):
            rand_aggr_losses = []
            greedy_losses = []
            sample_losses = []
            # batch is a length-4 list, each element is a tensor of shape (batch_size, 3, width, height)
            batch_size = batch[0].shape[0]
            cat_batch = torch.cat(batch, 0)
            if self.params.gpus > 0:
                cat_batch = cat_batch.to(self.device)
            cat_slots = self.model.forward(cat_batch, slots_only=True)#.unsqueeze(1)
            slots_A, slots_B, slots_C, slots_D = torch.split(cat_slots, batch_size, 0)

            batch_size, num_slots, slot_size = slots_A.shape

            # the full set of possible permutation might be too large ((7!)^3 for 7 slots...)
            # below implement three different approximation
            # 1. random projection to a high-dim space and then aggregate
            if not self.random_projection_init:
                self.random_projection = torch.rand(slot_size, slot_size*128)
                self.random_projection_init = True
                if self.params.gpus > 0:
                    self.random_projection = self.random_projection.to(self.device)
            proj_A = torch.matmul(slots_A, self.random_projection).mean(dim=-2) # average to make the scale invarint to slot number
            proj_B = torch.matmul(slots_B, self.random_projection).mean(dim=-2)
            proj_C = torch.matmul(slots_C, self.random_projection).mean(dim=-2)
            proj_D = torch.matmul(slots_D, self.random_projection).mean(dim=-2)
            rand_aggr_loss = torch.square(proj_A-proj_B+proj_C-proj_D).mean(dim=-1)
            rand_aggr_losses.append(rand_aggr_loss.squeeze(-1))

            # 2. greedy assignment regardless of re-assignment
            # TODO: check if there is a trivial solution to this assignment
            ext_A = slots_A.view(batch_size, num_slots, 1, 1, 1, slot_size)
            ext_B = slots_B.view(batch_size, 1, num_slots, 1, 1, slot_size)
            ext_C = slots_C.view(batch_size, 1, 1, num_slots, 1, slot_size)
            ext_D = slots_D.view(batch_size, 1, 1, 1, num_slots, slot_size)
            greedy_criterion = torch.square(ext_A-ext_B+ext_C-ext_D).sum(dim=-1)
            # backtrace for greedy matching (3 times)
            greedy_criterion, _ = greedy_criterion.min(-1)
            greedy_criterion, _ = greedy_criterion.min(-1)
            greedy_criterion, _ = greedy_criterion.min(-1)

            greedy_loss = greedy_criterion.sum(dim=-1)/(num_slots*slot_size)
            greedy_losses.append(greedy_loss)

            # 3. sampling based approximation
            slots_A = slots_A.repeat(sample_size, 1, 1)
            slots_B = slots_B.repeat(sample_size, 1, 1)
            slots_C = slots_C.repeat(sample_size, 1, 1)
            slots_D = slots_D.repeat(sample_size, 1, 1)

            # batch random permutation of slots https://discuss.pytorch.org/t/batch-version-of-torch-randperm/111121/3
            emb_A =slots_A.view(batch_size*sample_size, -1)
            emb_B =slots_B[torch.arange(slots_B.shape[0]).unsqueeze(-1), batch_rand_perm[:batch_size*sample_size]].view(batch_size*sample_size, -1)
            emb_C =slots_C[torch.arange(slots_C.shape[0]).unsqueeze(-1), batch_rand_perm[batch_size*sample_size:2*batch_size*sample_size]].view(batch_size*sample_size, -1)
            emb_D =slots_D[torch.arange(slots_D.shape[0]).unsqueeze(-1), batch_rand_perm[2*batch_size*sample_size:3*batch_size*sample_size]].view(batch_size*sample_size, -1)
            sample_loss = emb_A-emb_B+emb_C-emb_D
            sample_loss = torch.stack(torch.split(sample_loss, batch_size, 0), 1)
            sample_loss = torch.square(sample_loss).mean(dim=-1)
            sample_loss, _ = torch.min(sample_loss, 1)
            sample_losses.append(sample_loss)

            return rand_aggr_losses, greedy_losses, sample_losses

        for batch in odl:
            rand_aggr_losses, greedy_losses, sample_losses = compute_test_losses(batch)
            obj_rand_aggr_losses_epoch += rand_aggr_losses
            obj_greedy_losses_epoch += greedy_losses
            obj_sample_losses_epoch += sample_losses

        for batch in adl:
            rand_aggr_losses, greedy_losses, sample_losses = compute_test_losses(batch)
            attr_rand_aggr_losses_epoch += rand_aggr_losses
            attr_greedy_losses_epoch += greedy_losses
            attr_sample_losses_epoch += sample_losses

        avg_obj_aggr_loss = torch.cat(obj_rand_aggr_losses_epoch, 0).mean()
        avg_obj_greedy_loss = torch.cat(obj_greedy_losses_epoch, 0).mean()
        avg_obj_sample_loss = torch.cat(obj_sample_losses_epoch, 0).mean()
        avg_attr_aggr_loss = torch.cat(attr_rand_aggr_losses_epoch, 0).mean()
        avg_attr_greedy_loss = torch.cat(attr_greedy_losses_epoch, 0).mean()
        avg_attr_sample_loss = torch.cat(attr_sample_losses_epoch, 0).mean()
        logs = {
            "avg_val_loss": avg_loss,
            "avg_obj_aggr_loss": avg_obj_aggr_loss,
            "avg_obj_greedy_loss": avg_obj_greedy_loss,
            "avg_obj_sample_loss": avg_obj_sample_loss,
            "avg_attr_aggr_loss": avg_attr_aggr_loss,
            "avg_attr_greedy_loss": avg_attr_greedy_loss,
            "avg_attr_sample_loss": avg_attr_sample_loss,
        }
        self.log_dict(logs, sync_dist=True)

        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        warmup_steps_pct = self.params.warmup_steps_pct
        decay_steps_pct = self.params.decay_steps_pct
        total_steps = self.params.max_epochs * len(self.datamodule.train_dataloader())

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < total_steps
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
