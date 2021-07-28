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
        dl = self.datamodule.test_dataloader()
        test_losses = []
        for batch in dl:
            # batch is a length-4 list, each element is a tensor of shape (batch_size, 3, width, height)
            batch_size = batch[0].shape[0]
            cat_batch = torch.cat(batch, 0)
            if self.params.gpus > 0:
                cat_batch = cat_batch.to(self.device)
            cat_slots = self.model.forward(cat_batch, slots_only=True).unsqueeze(1)
            slots_A, slots_B, slots_C, slots_D = torch.split(cat_slots, batch_size, 0)

            batch_size, _, num_slots, slot_size = slots_A.shape
            # the full set of possible permutation might be too large ((7!)^3 for 7 slots...)
            # I sample 100000 permutation tuples as an approximation
            emb_losses = []
            # TODO: make it parallel
            # for _ in range(10):
            #     rand = torch.rand(100*4, num_slots)
            #     batch_rand_perm = rand.argsort(dim=1)
            #     if self.params.gpus > 0:
            #         batch_rand_perm = batch_rand_perm.to(self.device)
            #     slots_A = slots_A.repeat(1, 100, 1, 1)
            #     slots_B = slots_B.repeat(1, 100, 1, 1)
            #     slots_C = slots_C.repeat(1, 100, 1, 1)
            #     slots_D = slots_D.repeat(1, 100, 1, 1)

            #     emb_A =slots_A[batch_rand_perm[:100]].view(batch_size, 1, -1)
            #     emb_B =slots_B[batch_rand_perm[100:200]].view(batch_size, 1, -1)
            #     emb_C =slots_C[batch_rand_perm[200:300]].view(batch_size, 1, -1)
            #     emb_D =slots_D[batch_rand_perm[300:400]].view(batch_size, 1, -1)
            #     emb_loss = torch.square(emb_A-emb_B+emb_C-emb_D).mean(dim=2)
            #     emb_loss, _ = torch.min(emb_loss, 1)
            #     print(emb_loss.data.tolist())
            #     emb_losses.append(emb_loss)
            # test_loss = torch.cat(emb_losses, 1)
            # test_loss, _ = torch.min(emb_loss, 1)

            for _ in range(1, 10000):

                perm_A = torch.randperm(num_slots)
                emb_A = slots_A[:, :, perm_A].view(batch_size, 1, -1)
                perm_B = torch.randperm(num_slots)
                emb_B = slots_B[:, :, perm_B].view(batch_size, 1, -1)
                perm_C = torch.randperm(num_slots)
                emb_C = slots_C[:,:, perm_C].view(batch_size, 1, -1)
                perm_D = torch.randperm(num_slots)
                emb_D = slots_D[:, :, perm_D].view(batch_size, 1, -1)

                emb_loss = torch.square(emb_A-emb_B+emb_C-emb_D).mean(dim=2)
                emb_losses.append(emb_loss)
            
            test_loss = torch.cat(emb_losses, 1)
            test_loss, _ = torch.min(test_loss, 1)
            test_losses.append(test_loss)
        avg_test_loss = torch.cat(test_losses, 0).mean()
        logs = {
            "avg_val_loss": avg_loss,
            "avg_test_loss": avg_test_loss,
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
