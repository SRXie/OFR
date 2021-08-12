import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils
from torchvision.transforms import transforms
from PIL import ImageDraw, ImageFont
from datetime import datetime


from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import Tensor
from slot_attention.utils import to_rgb_from_tensor, to_tensor_from_rgb
from slot_attention.utils import compute_cos_distance, compute_rank_correlation
from slot_attention.utils import batched_index_select
from slot_attention.utils import compute_greedy_loss, compute_pseudo_greedy_loss
from slot_attention.utils import swap_bg_slot_back


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
        batch = next(iter(dl))
        batch = torch.stack([b[1] for b in batch], 0)
        if self.params.gpus > 0:
            batch = batch.to(self.device)

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
            masked_recons = recons * masks + (1 - masks)
            masked_recons[:,:,0,:,:] = masked_recons[:,:,0,:,:]+attn
            masked_recons = to_rgb_from_tensor(masked_recons)
            for i in range(masked_recons.shape[0]):
                for j in range(masked_recons.shape[1]):
                    img = transforms.ToPILImage()(masked_recons[i,j])
                    draw = ImageDraw.Draw(img)
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
                    pixel_text = "attn: "+str(pixel_dup_idx[i,j].item())+" - {:.4f}".format(pixel_dup_sim[i,j].item())
                    feature_text = "feat: "+str(feature_dup_idx[i,j].item())+" - {:.4f}".format(feature_dup_sim[i,j].item())
                    draw.text((4,0), pixel_text, (0, 0, 0), font=font)
                    draw.text((4,55), feature_text, (0, 0, 0), font=font)
                    img = transforms.ToTensor()(img)
                    img = to_tensor_from_rgb(img)
                    masked_recons[i,j] = img
            return masked_recons, attn

        recon_combined, recons, masks, slots, attns, recon_combined_nodup, recons_nodup, masks_nodup, slots_nodup = self.model.forward(batch, dup_threshold=self.params.dup_threshold)
        # throw background slot back
        cat_indices = swap_bg_slot_back(attns)
        recons = batched_index_select(recons, 1, cat_indices)
        masks = batched_index_select(masks, 1, cat_indices)
        slots = batched_index_select(slots, 1, cat_indices)
        attns = batched_index_select(attns, 2, cat_indices)
        recons_nodup = batched_index_select(recons_nodup, 1, cat_indices)
        masks_nodup = batched_index_select(masks_nodup, 1, cat_indices)
        slots_nodup = batched_index_select(slots_nodup, 1, cat_indices)

        # reorder with matching
        cat_indices = compute_greedy_loss(slots, [])
        recons_perm = batched_index_select(recons, 1, cat_indices)
        masks_perm = batched_index_select(masks, 1, cat_indices)
        slots_perm = batched_index_select(slots, 1, cat_indices)
        attns_perm = batched_index_select(attns, 2, cat_indices)
        masked_recons_perm, attn_perm = captioned_masked_recons(recons_perm, masks_perm, slots_perm, attns_perm)

        cat_indices_nodup = compute_greedy_loss(slots_nodup, [])
        recons_perm_nodup = batched_index_select(recons_nodup, 1, cat_indices_nodup)
        masks_perm_nodup = batched_index_select(masks_nodup, 1, cat_indices_nodup)
        slots_perm_nodup = batched_index_select(slots_nodup, 1, cat_indices_nodup)
        attns_perm_nodup = batched_index_select(attns, 2, cat_indices_nodup)
        masked_recons_perm_nodup, attn_perm_nodup = captioned_masked_recons(recons_perm_nodup, masks_perm_nodup, slots_perm_nodup, attns_perm_nodup)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    torch.cat([batch.unsqueeze(1), batch.unsqueeze(1)], dim=0),  # original images
                    torch.cat([recon_combined.unsqueeze(1),recon_combined_nodup.unsqueeze(1)], dim=0),  # reconstructions
                    torch.cat([masked_recons_perm, masked_recons_perm_nodup], dim=0),  # each slot
                    torch.cat([recons_perm, recons_perm_nodup], dim=0),
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
        val_loss = self.model.loss_function(batch)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # Algebra Test starts here
        odl = self.datamodule.obj_test_dataloader()
        adl = self.datamodule.attr_test_dataloader()
        sample_size = 10000
        obj_greedy_losses, attr_greedy_losses = [], []
        obj_greedy_losses_nodup, attr_greedy_losses_nodup = [], []
        obj_pd_greedy_losses, attr_pd_greedy_losses = [], []
        obj_pd_greedy_losses_nodup, attr_pd_greedy_losses_nodup = [], []

        # rand = torch.rand(self.params.batch_size*sample_size*3, self.params.num_slots)
        # batch_rand_perm = rand.argsort(dim=1)
        # del rand
        # if self.params.gpus > 0:
            # batch_rand_perm = batch_rand_perm.to(self.device)

        def compute_test_losses(dataloader, losses, pseudo_losses, losses_nodup, pseudo_losses_nodup, dup_threshold=None):

            b_prev = datetime.now()
            for _ in range(1000):
                batch = next(iter(dataloader))
                # print("load data:", datetime.now()-b_prev)
                # sample_losses = []
                # batch is a length-4 list, each element is a tensor of shape (batch_size, 3, width, height)
                batch_size = batch[0].shape[0]
                cat_batch = torch.cat(batch, 0)
                if self.params.gpus > 0:
                    cat_batch = cat_batch.to(self.device)
                cat_slots, cat_attns, cat_slots_nodup = self.model.forward(cat_batch, slots_only=True, dup_threshold=dup_threshold)

                cat_indices = swap_bg_slot_back(cat_attns)
                cat_slots = batched_index_select(cat_slots, 1, cat_indices)
                cat_slots_nodup = batched_index_select(cat_slots_nodup, 1, cat_indices)

                compute_greedy_loss(cat_slots, losses)
                compute_greedy_loss(cat_slots_nodup, losses_nodup)

                compute_pseudo_greedy_loss(cat_slots, pseudo_losses)
                compute_pseudo_greedy_loss(cat_slots_nodup, pseudo_losses_nodup)

                # slots_A = slots_A.repeat(sample_size, 1, 1)
                # slots_B = slots_B.repeat(sample_size, 1, 1)
                # slots_C = slots_C.repeat(sample_size, 1, 1)
                # slots_D = slots_D.repeat(sample_size, 1, 1)

                # # batch random permutation of slots https://discuss.pytorch.org/t/batch-version-of-torch-randperm/111121/3
                # emb_A =slots_A.view(batch_size*sample_size, -1)
                # emb_B =slots_B[torch.arange(slots_B.shape[0]).unsqueeze(-1), batch_rand_perm[:batch_size*sample_size]].view(batch_size*sample_size, -1)
                # emb_C =slots_C[torch.arange(slots_C.shape[0]).unsqueeze(-1), batch_rand_perm[batch_size*sample_size:2*batch_size*sample_size]].view(batch_size*sample_size, -1)
                # emb_D =slots_D[torch.arange(slots_D.shape[0]).unsqueeze(-1), batch_rand_perm[2*batch_size*sample_size:3*batch_size*sample_size]].view(batch_size*sample_size, -1)
                # sample_loss = emb_A-emb_B+emb_C-emb_D
                # sample_loss = torch.stack(torch.split(sample_loss, batch_size, 0), 1)
                # sample_loss = torch.square(sample_loss).mean(dim=-1)
                # sample_loss, _ = torch.min(sample_loss, 1)
                # sample_losses.appiend(sample_loss)
                # print("batch time:", datetime.now()-b_prev)
                # b_prev = datetime.now()

        compute_test_losses(odl, obj_greedy_losses, obj_pd_greedy_losses, obj_greedy_losses_nodup, obj_pd_greedy_losses_nodup, dup_threshold=self.params.dup_threshold)
        compute_test_losses(adl, attr_greedy_losses, attr_pd_greedy_losses, attr_greedy_losses_nodup, attr_pd_greedy_losses_nodup, dup_threshold=self.params.dup_threshold)

        avg_obj_greedy_loss = torch.cat(obj_greedy_losses, 0).mean()
        avg_attr_greedy_loss = torch.cat(attr_greedy_losses, 0).mean()

        avg_obj_greedy_loss_nodup = torch.cat(obj_greedy_losses_nodup, 0).mean()
        avg_attr_greedy_loss_nodup = torch.cat(attr_greedy_losses_nodup, 0).mean()

        avg_obj_pd_greedy_loss = torch.cat(obj_pd_greedy_losses, 0).mean()
        avg_attr_pd_greedy_loss = torch.cat(attr_pd_greedy_losses, 0).mean()
        avg_obj_pd_greedy_loss_nodup = torch.cat(obj_pd_greedy_losses_nodup, 0).mean()
        avg_attr_pd_greedy_loss_nodup = torch.cat(attr_pd_greedy_losses_nodup, 0).mean()

        logs = {
            "avg_val_loss": avg_loss,
            "avg_obj_greedy_loss": avg_obj_greedy_loss,
            "avg_attr_greedy_loss": avg_attr_greedy_loss,
            "avg_obj_greedy_loss_nodup": avg_obj_greedy_loss_nodup,
            "avg_attr_greedy_loss_nodup": avg_attr_greedy_loss_nodup,
            "avg_obj_pseudo_greedy_loss": avg_obj_pd_greedy_loss,
            "avg_attr_pseudo_greedy_loss": avg_attr_pd_greedy_loss,
            "avg_obj_pseudo_greedy_loss_nodup": avg_obj_pd_greedy_loss_nodup,
            "avg_attr_pseudo_greedy_loss_nodup": avg_attr_pd_greedy_loss_nodup,
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
