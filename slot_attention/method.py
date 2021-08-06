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
            recons[:,:,0,:,:] = recons[:,:,0,:,:]+attn
            masked_recons = recons * masks + (1 - masks)
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
        masked_recons, attn = captioned_masked_recons(recons, masks, slots, attns)
        masked_recons_nodup, attn_nodup = captioned_masked_recons(recons_nodup, masks_nodup, slots_nodup, attns)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        def interleave_stack(x, y):
            return torch.stack((x, y), dim=1).view(2*x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    interleave_stack(batch.unsqueeze(1), batch.unsqueeze(1)), #torch.cat([batch.unsqueeze(1), batch.unsqueeze(1)], dim=0),  # original images
                    interleave_stack(recon_combined.unsqueeze(1),recon_combined_nodup.unsqueeze(1)),  # reconstructions
                    interleave_stack(masked_recons, masked_recons_nodup),  # each slot
                    interleave_stack(attn.unsqueeze(2).repeat(1,1,3,1,1), attn_nodup.unsqueeze(2).repeat(1,1,3,1,1)),
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

            def _compute_pseudo_greedy_loss(cat_slots, losses):
                slots_A, slots_B, slots_C, slots_D = torch.split(cat_slots, batch_size, 0)
                # greedy assignment regardless of re-assignment
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
                losses.append(greedy_loss)

            def _compute_greedy_loss(cat_slots, losses):
                slots_A, slots_B, slots_C, slots_D = torch.split(cat_slots, batch_size, 0)
                # greedy assignment without multi-assignment
                greedy_loss = torch.zeros(batch_size)
                if self.params.gpus > 0:
                    greedy_loss = greedy_loss.to(self.device)
                for i in range(num_slots):
                    # TODO: check if there is a trivial solution to this assignment
                    ext_A = slots_A.view(batch_size, num_slots-i, 1, 1, 1, slot_size)
                    ext_B = slots_B.view(batch_size, 1, num_slots-i, 1, 1, slot_size)
                    ext_C = slots_C.view(batch_size, 1, 1, num_slots-i, 1, slot_size)
                    ext_D = slots_D.view(batch_size, 1, 1, 1, num_slots-i, slot_size)
                    greedy_criterion = torch.square(ext_A-ext_B+ext_C-ext_D).sum(dim=-1)
                    # backtrace for greedy matching (4 times)
                    greedy_criterion, indices_D = greedy_criterion.min(-1)
                    greedy_criterion, indices_C = greedy_criterion.min(-1)
                    greedy_criterion, indices_B = greedy_criterion.min(-1)
                    greedy_criterion, indices_A = greedy_criterion.min(-1)
                    greedy_loss+=greedy_criterion


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
                    if self.params.gpus>0:
                        replace = replace.to(self.device)
                    index_cat = torch.cat([index_A, index_B, index_C, index_D], dim=0)
                    slots_cat = torch.cat([slots_A, slots_B, slots_C, slots_D], dim=0)

                    replace = replace.scatter(1, index_cat, True)
                    replace = replace.unsqueeze(-1).repeat(1, 1, slot_size)
                    slots_cat = torch.where(replace, slots_cat[:,-1,:].unsqueeze(1).repeat(1, num_slots-i, 1), slots_cat)[:,:-1,:]
                    slots_A, slots_B, slots_C, slots_D = torch.split(slots_cat, batch_size, 0)

                greedy_loss = greedy_loss/(num_slots*slot_size)
                losses.append(greedy_loss)

            b_prev = datetime.now()
            for batch in dataloader:
                print("load data:", datetime.now()-b_prev)
                # rand_aggr_losses = []
                # greedy_losses = []
                # sample_losses = []
                # batch is a length-4 list, each element is a tensor of shape (batch_size, 3, width, height)
                batch_size = batch[0].shape[0]
                cat_batch = torch.cat(batch, 0)
                if self.params.gpus > 0:
                    cat_batch = cat_batch.to(self.device)
                cat_slots, cat_attns, cat_slots_nodup = self.model.forward(cat_batch, slots_only=True, dup_threshold=dup_threshold)

                _, num_slots, slot_size = cat_slots.shape

                _compute_greedy_loss(cat_slots, losses)
                _compute_greedy_loss(cat_slots_nodup, losses_nodup)

                _compute_pseudo_greedy_loss(cat_slots, pseudo_losses)
                _compute_pseudo_greedy_loss(cat_slots_nodup, pseudo_losses_nodup)
                # # cat_attns have shape (4*batch_size, H*W, num_slots)
                # prev = datetime.now()
                # # calculate intra-image slots similarity in the pixel space:
                # cos_dis_pixel = compute_cos_distance(cat_attns.permute(0,2,1))

                # # calculate intra-image slots similarity in the feature space:
                # cos_dis_feature = compute_cos_distance(cat_slots)

                # kendall_tau = compute_rank_correlation(cos_dis_pixel.view(-1,num_slots), cos_dis_feature.view(-1,num_slots))
                # kendall_taus.append(kendall_tau)
                # print("similarity time:", datetime.now()-prev)

                # slots_A, slots_B, slots_C, slots_D = torch.split(cat_slots, batch_size, 0)

                # the full set of possible permutation might be too large ((7!)^3 for 7 slots...)
                # below implement three different approximation
                # 1. random projection to a high-dim space and then aggregate
                # if not self.random_projection_init:
                #     self.random_projection = torch.rand(slot_size, slot_size*128)
                #     self.random_projection_init = True
                #     if self.params.gpus > 0:
                #         self.random_projection = self.random_projection.to(self.device)
                # proj_A = torch.matmul(slots_A, self.random_projection).mean(dim=-2) # average to make the scale invarint to slot number
                # proj_B = torch.matmul(slots_B, self.random_projection).mean(dim=-2)
                # proj_C = torch.matmul(slots_C, self.random_projection).mean(dim=-2)
                # proj_D = torch.matmul(slots_D, self.random_projection).mean(dim=-2)
                # rand_aggr_loss = torch.square(proj_A-proj_B+proj_C-proj_D).mean(dim=-1)
                # rand_aggr_losses.append(rand_aggr_loss.squeeze(-1))

                # # 2. greedy assignment regardless of re-assignment
                # # TODO: check if there is a trivial solution to this assignment
                # ext_A = slots_A.view(batch_size, num_slots, 1, 1, 1, slot_size)
                # ext_B = slots_B.view(batch_size, 1, num_slots, 1, 1, slot_size)
                # ext_C = slots_C.view(batch_size, 1, 1, num_slots, 1, slot_size)
                # ext_D = slots_D.view(batch_size, 1, 1, 1, num_slots, slot_size)
                # greedy_criterion = torch.square(ext_A-ext_B+ext_C-ext_D).sum(dim=-1)
                # # backtrace for greedy matching (3 times)
                # greedy_criterion, _ = greedy_criterion.min(-1)
                # greedy_criterion, _ = greedy_criterion.min(-1)
                # greedy_criterion, _ = greedy_criterion.min(-1)

                # greedy_loss = greedy_criterion.sum(dim=-1)/(num_slots*slot_size)
                # losses.append(greedy_loss)

                # 3. sampling based approximation
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
                print("batch time:", datetime.now()-b_prev)
                b_prev = datetime.now()

        compute_test_losses(odl, obj_greedy_losses, obj_pd_greedy_losses, obj_greedy_losses_nodup, obj_pd_greedy_losses_nodup, dup_threshold=self.params.dup_threshold)
        compute_test_losses(adl, attr_greedy_losses, attr_pd_greedy_losses, attr_greedy_losses_nodup, attr_pd_greedy_losses_nodup, dup_threshold=self.params.dup_threshold)

        avg_obj_greedy_loss = torch.cat(obj_greedy_losses, 0).mean()
        avg_attr_greedy_loss = torch.cat(attr_greedy_losses, 0).mean()
        # avg_obj_kendall_tau = torch.stack(obj_kendall_taus).mean()
        # avg_attr_kendall_tau = torch.stack(attr_kendall_taus).mean()

        avg_obj_greedy_loss_nodup = torch.cat(obj_greedy_losses, 0).mean()
        avg_attr_greedy_loss_nodup = torch.cat(attr_greedy_losses, 0).mean()

        avg_obj_pd_greedy_loss = torch.cat(obj_pd_greedy_losses, 0).mean()
        avg_attr_pd_greedy_loss = torch.cat(attr_pd_greedy_losses, 0).mean()
        avg_obj_pd_greedy_loss_nodup = torch.cat(obj_pd_greedy_losses, 0).mean()
        avg_attr_pd_greedy_loss_nodup = torch.cat(attr_pd_greedy_losses, 0).mean()

        if self.params.gpus > 0:
            self.model.blank_slot = self.model.blank_slot.to(self.device)
            self.model.slots_mu = self.model.slots_mu.to(self.device)
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
