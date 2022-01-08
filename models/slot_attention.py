from typing import Tuple

import math
import itertools
import torch
from torch import nn
from torch.nn import functional as F

from utils import Tensor
from utils import assert_shape
from utils import build_grid
from utils import conv_transpose_out_shape
from utils import compute_cos_distance
from utils import batched_index_select
from utils import compute_mask_ari
from utils import to_rgb_from_tensor
from utils import compute_corr_coef
from utils import compute_greedy_loss, compute_pseudo_greedy_loss


class SlotAttention(nn.Module):
    def __init__(self, in_features, num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )

    def forward(self, inputs: Tensor):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots_init = torch.randn((batch_size, self.num_slots, self.slot_size))
        slots_init = slots_init.type_as(inputs)
        slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init
        attns = None
        attns_init = False

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            assert_shape(q.size(), (batch_size, self.num_slots, self.slot_size))

            attn_norm_factor = self.slot_size ** -0.5
            attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
            attn = F.softmax(attn_logits, dim=-1)
            if not torch.is_tensor(attns):
                attns = attn.clone().detach()
            else:
                attns = attns + attn
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn.size(), (batch_size, num_inputs, self.num_slots))

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates.size(), (batch_size, self.num_slots, self.slot_size))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
            slots = slots + self.mlp(self.norm_mlp(slots))
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))

        return slots, attns/self.num_iterations


class SlotAttentionModel(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations,
        in_channels: int = 3,
        kernel_size: int = 5,
        slot_size: int = 64,
        hidden_dims: Tuple[int, ...] = (64, 64, 64, 64), #delete one entry for 128 -> 64
        decoder_resolution: Tuple[int, int] = (8, 8),
        empty_cache=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.slot_size = slot_size
        self.empty_cache = empty_cache
        self.hidden_dims = hidden_dims
        self.decoder_resolution = decoder_resolution
        self.out_features = self.hidden_dims[-1]
        self.layer_norm = nn.LayerNorm(self.out_features)

        modules = []
        channels = self.in_channels
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2,
                    ),
                    nn.LeakyReLU(),
                )
            )
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.LeakyReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        # Build Decoder
        modules = []

        in_size = decoder_resolution[0]
        out_size = in_size

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)

        assert_shape(
            resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
        )

        # same convolutions
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.out_features, self.out_features, kernel_size=5, stride=1, padding=2, output_padding=0,
                ),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(self.out_features, 4, kernel_size=3, stride=1, padding=1, output_padding=0,),
            )
        )

        assert_shape(resolution, (out_size, out_size), message="")

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, self.decoder_resolution)

        self.slot_attention = SlotAttention(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=128,
        )

        self.slots_mu = self.slot_attention.slots_mu
        self.slots_log_sigma = self.slot_attention.slots_log_sigma
        self.blank_slot = None

    def forward(self, x, slots_only=False, dup_threshold=None, viz=False):
        if self.empty_cache:
            torch.cuda.empty_cache()

        batch_size, num_channels, height, width = x.shape
        encoder_out = self.encoder(x)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, filter_size, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, filter_size, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        # SR: Add Layer Norm
        encoder_out = self.layer_norm(encoder_out)
        encoder_out = self.encoder_out_layer(encoder_out)
        #  `encoder_out` has shape: [batch_size, height*width, filter_size]

        slots, attn = self.slot_attention(encoder_out)
        assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
        # `slots` has shape: [batch_size, num_slots, slot_size].
        batch_size, num_slots, slot_size = slots.shape

        # # to keep the one with largest attention mass in dup removal, sort slots by max-pooled attention mass
        # attn_mass = attn.permute(0,2,1).clone()
        # attn_mass = torch.where(attn_mass>=attn_mass.max(dim=1)[0].unsqueeze(1).repeat(1,num_slots,1), attn_mass, torch.zeros_like(attn_mass)).sum(-1)
        # idx = torch.argsort(attn_mass.detach(), dim=1, descending=True)
        # slots = batched_index_select(slots, 1, idx)
        # attn = batched_index_select(attn, 2, idx)

        slots_nodup = slots.clone()
        if dup_threshold:
            cos_dis_pixel = compute_cos_distance(attn.permute(0,2,1)) # to have shape (batch_size, num_slot, emb_size)
            cos_dis_feature = compute_cos_distance(slots)
            cos_dis_min = torch.min(cos_dis_pixel, cos_dis_feature)
            duplicated = cos_dis_feature < dup_threshold
            # we only need the upper triangle
            duplicated = torch.triu(duplicated, diagonal=1)
            duplicated = torch.sum(duplicated, dim=1)
            duplicated_index = torch.nonzero(duplicated, as_tuple=True)
            # get blank slots
            blank_slots = slots.view(-1, slot_size).mean(0).unsqueeze(0)
            # fill in deuplicated slots with blank slots
            slots_nodup[duplicated_index[0], duplicated_index[1]] = blank_slots

        if slots_only:
            return slots, attn, slots_nodup

        # slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
        if dup_threshold:
            slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
            slots_nodup = slots_nodup.view(batch_size * num_slots, slot_size, 1, 1)
            slots_cat = torch.cat([slots, slots_nodup])
            batch_size = batch_size*2
        else:
            slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
            slots_cat = slots
        decoder_in = slots_cat.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        assert_shape(out.size(), (batch_size * num_slots, num_channels + 1, height, width))

        out = out.view(batch_size, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]
        masks = out[:, :, -1:, :, :]
        if dup_threshold:
            unnormalized_masks_nodup = masks[:batch_size//2].clone()
        masks = F.softmax(masks, dim=1)

        recon_combined = torch.sum(recons * masks, dim=1)

        if dup_threshold:
            batch_size = batch_size//2
            slots = slots.view(batch_size, num_slots, slot_size)
            recons, recons_nodup = torch.split(recons, batch_size, 0)
            masks, masks_nodup = torch.split(masks, batch_size, 0)
            recon_combined, recon_combined_nodup = torch.split(recon_combined, batch_size, 0)
            slots_nodup = slots_nodup.view(batch_size, num_slots, slot_size)

            masks_nodup_mass = masks_nodup.view(batch_size, num_slots, -1)
            masks_nodup_mass = torch.where(masks_nodup_mass>=masks_nodup_mass.max(dim=1)[0].unsqueeze(1).repeat(1,recons.shape[1],1), masks_nodup_mass, torch.zeros_like(masks_nodup_mass)).sum(-1)
            invisible_index = torch.nonzero(masks_nodup_mass==0.0, as_tuple=True)
            slots_nodup[invisible_index[0], invisible_index[1]] = blank_slots

            unnormalized_masks_nodup[duplicated_index[0], duplicated_index[1]] = -1000000.0*torch.ones_like(masks_nodup_mass[0,0])
            masks_nodup = F.softmax(unnormalized_masks_nodup, dim=1)
            recon_combined_nodup = torch.sum(recons_nodup * masks_nodup, dim=1)

            if viz:
                # Here we reconstruct D'
                batch_size = batch_size//4
                _, cat_indices = compute_greedy_loss(slots_nodup) #compute_pseudo_greedy_loss(slots, [])
                slots_nodup = batched_index_select(slots_nodup, 1, cat_indices) # batched_index_select(slots, 1, cat_indices)
                recons_nodup = batched_index_select(recons_nodup, 1, cat_indices)
                masks_nodup = batched_index_select(masks_nodup, 1, cat_indices)
                slots_D_prime=slots_nodup[:batch_size]-slots_nodup[batch_size:2*batch_size]+slots_nodup[2*batch_size:3*batch_size]
                slots_D_prime = slots_D_prime.view(batch_size * num_slots, slot_size, 1, 1)

                decoder_in = slots_D_prime.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])

                out_D_prime = self.decoder_pos_embedding(decoder_in)
                out_D_prime = self.decoder(out_D_prime)
                # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
                assert_shape(out_D_prime.size(), (batch_size * num_slots, num_channels + 1, height, width))

                out_D_prime = out_D_prime.view(batch_size, num_slots, num_channels + 1, height, width)
                recons_D_prime = out_D_prime[:, :, :num_channels, :, :]
                masks_D_prime = out_D_prime[:, :, -1:, :, :]
                slots_D_prime = slots_D_prime.view(batch_size, num_slots, slot_size)
                # Here we mask the de-duplicated slots in generation
                detect_blank_slots = slots_D_prime.view(batch_size, num_slots, -1)
                detect_blank_slots = ((detect_blank_slots - blank_slots.unsqueeze(0)).sum(-1)==0.0).nonzero(as_tuple=True)
                masks_D_prime[detect_blank_slots[0], detect_blank_slots[1]] = -10000.0*torch.ones_like(masks_D_prime[0,0])
                masks_D_prime = F.softmax(masks_D_prime, dim=1)
                recon_combined_D_prime = torch.sum(recons_D_prime * masks_D_prime, dim=1)

                recon_combined_nodup = torch.cat(list(recon_combined_nodup.split(batch_size, 0))+[recon_combined_D_prime], 0)
                recons_nodup = torch.cat(list(recons_nodup.split(batch_size, 0))+[recons_D_prime], 0)
                masks_nodup = torch.cat(list(masks_nodup.split(batch_size, 0))+[masks_D_prime], 0)
                slots_nodup = torch.cat(list(slots_nodup.split(batch_size, 0))+[slots_D_prime], 0)
            return recon_combined, recons, masks, slots, attn, recon_combined_nodup, recons_nodup, masks_nodup, slots_nodup
        else:
            # slots = slots.view(batch_size, num_slots, slot_size)
            return recon_combined, recons, masks, slots, attn

    def loss_function(self, input, mask_gt=None, schema_gt=None):
        recon_combined, recons, masks, slots, attn = self.forward(input)
        loss = F.mse_loss(recon_combined, input)

        if not mask_gt:
            return {
                "loss": loss,
            }
        else:
            # compute ARI with mask gt
            # (batch_size, num_slots, 1, H, W) to (batch_size, num_slots, H, W)
            pred_mask = masks.squeeze(2)

            batch_size, num_slots, H, W = pred_mask.size()
            mask_gt = to_rgb_from_tensor(torch.stack(mask_gt, 1)[:,:,0,:,:])
            assert_shape(mask_gt.shape, pred_mask.shape)
            # index shape (batch_size, H, W)
            index = torch.argmax(pred_mask, dim=1)
            # get binarized masks (batch_size, , H, W)
            pred_mask = torch.zeros_like(pred_mask)
            pred_mask[torch.arange(batch_size)[:, None, None], index, torch.arange(H)[None, :, None], torch.arange(W)[None, None, :]] = 1.0

            mask_aris = None
            for b in range(batch_size):
                mask_ari = compute_mask_ari(mask_gt[b].detach().cpu(), pred_mask[b].detach().cpu())
                if not mask_aris:
                    mask_aris = mask_ari
                else:
                    mask_aris += mask_ari
            mask_ari = mask_aris/batch_size

            # # Here we start to calculate the correlation between the distance in the schema and the l2 distance in the slot features

            # perm_num = math.factorial(num_slots)
            # perm_idx = list(itertools.permutations([a for a in range(num_slots)]))
            # perm_idx = torch.stack([torch.Tensor(idx) for idx in perm_idx], dim=0).long()
            # perm_idx = perm_idx.repeat(batch_size, 1)

            # # First we get the best-matched distance for all schema pairs in this batch
            # schema_gt = schema_gt[:, :num_slots]
            # schema_a = torch.reshape(schema_gt, (batch_size, 1, 1, num_slots, -1))
            # schema_a = schema_a.repeat(1, batch_size, perm_num, 1, 1)
            # schema_b = schema_gt.unsqueeze(1).repeat(1, perm_num, 1, 1).view(batch_size*perm_num, num_slots, -1)
            # schema_b = schema_b[torch.arange(batch_size*perm_num).unsqueeze(-1), perm_idx]
            # schema_b = schema_b.view(1, batch_size, perm_num, num_slots, -1).repeat(batch_size, 1, 1, 1, 1)
            # schema_a_disc = torch.reshape(schema_a[:,:,:,:, :4], (batch_size, batch_size, perm_num, -1))
            # schema_b_disc = torch.reshape(schema_b[:,:,:,:, :4], (batch_size, batch_size, perm_num, -1))
            # schema_a_size = torch.reshape(schema_a[:,:,:,:, 0], (batch_size, batch_size, perm_num, -1))
            # schema_b_size = torch.reshape(schema_b[:,:,:,:, 0], (batch_size, batch_size, perm_num, -1))
            # schema_a_material = torch.reshape(schema_a[:,:,:,:, 1], (batch_size, batch_size, perm_num, -1))
            # schema_b_material = torch.reshape(schema_b[:,:,:,:, 1], (batch_size, batch_size, perm_num, -1))
            # schema_a_shape = torch.reshape(schema_a[:,:,:,:, 2], (batch_size, batch_size, perm_num, -1))
            # schema_b_shape = torch.reshape(schema_b[:,:,:,:, 2], (batch_size, batch_size, perm_num, -1))
            # schema_a_color = torch.reshape(schema_a[:,:,:,:, 3], (batch_size, batch_size, perm_num, -1))
            # schema_b_color = torch.reshape(schema_b[:,:,:,:, 3], (batch_size, batch_size, perm_num, -1))
            # schema_a_pos = torch.reshape(schema_a[:,:,:,:, 4:6], (batch_size, batch_size, perm_num, -1))/6.0
            # schema_b_pos = torch.reshape(schema_b[:,:,:,:, 4:6], (batch_size, batch_size, perm_num, -1))/6.0
            # schema_distance_disc = torch.where(schema_a_disc == schema_b_disc, 1.0, 0.0).sum(-1)/(num_slots)
            # schema_distance_disc, _ = schema_distance_disc.max(-1)
            # schema_distance_disc = 4.0 - schema_distance_disc
            # schema_distance_pos, idx_pos = torch.norm(schema_a_pos - schema_b_pos, p=2, dim=-1).min(-1)
            # schema_distance_size = torch.where(schema_a_size == schema_b_size, 1.0, 0.0).sum(-1)/(num_slots)
            # schema_distance_material = torch.where(schema_a_material == schema_b_material, 1.0, 0.0).sum(-1)/(num_slots)
            # schema_distance_shape = torch.where(schema_a_shape == schema_b_shape, 1.0, 0.0).sum(-1)/(num_slots)
            # schema_distance_color = torch.where(schema_a_color == schema_b_color, 1.0, 0.0).sum(-1)/(num_slots)
            # idx_pos = idx_pos.view(batch_size*batch_size, 1)
            # schema_distance_size = batched_index_select(schema_distance_size.view(batch_size*batch_size, perm_num), 1, idx_pos).view(batch_size, batch_size)
            # schema_distance_material = batched_index_select(schema_distance_material.view(batch_size*batch_size, perm_num), 1, idx_pos).view(batch_size, batch_size)
            # schema_distance_shape = batched_index_select(schema_distance_shape.view(batch_size*batch_size, perm_num), 1, idx_pos).view(batch_size, batch_size)
            # schema_distance_color = batched_index_select(schema_distance_color.view(batch_size*batch_size, perm_num), 1, idx_pos).view(batch_size, batch_size)

            # # get rid of the diagonal
            # schema_distance_disc = schema_distance_disc.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1).flatten()
            # schema_distance_pos = schema_distance_pos.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1).flatten()
            # schema_distance_size = schema_distance_size.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1).flatten()
            # schema_distance_material = schema_distance_material.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1).flatten()
            # schema_distance_shape = schema_distance_shape.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1).flatten()
            # schema_distance_color = schema_distance_color.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1).flatten()

            # # Then we get the best-matched l2 distance for all image pairs
            # slots_a = slots.view(batch_size, 1, 1, -1)
            # slots_a = slots_a.repeat(1, batch_size, perm_num, 1)
            # slots_b = slots.unsqueeze(1).repeat(1, perm_num, 1, 1).view(batch_size*perm_num, num_slots, -1)
            # slots_b = slots_b[torch.arange(batch_size*perm_num).unsqueeze(-1), perm_idx]
            # slots_b = slots_b.view(1, batch_size, perm_num, -1).repeat(batch_size, 1, 1, 1)
            # slots_distance, _ = torch.norm(slots_a - slots_b, p = 2, dim=-1).min(-1)
            # slots_distance = slots_distance/(num_slots*slots.shape[-1])
            # # get rid of the diagonal
            # slots_distance = slots_distance.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1).flatten()

            return {
                "loss": loss,
                "mask_ari": mask_ari,
                # "schema_distance_disc": schema_distance_disc,
                # "schema_distance_size": schema_distance_size,
                # "schema_distance_material": schema_distance_material,
                # "schema_distance_shape": schema_distance_shape,
                # "schema_distance_color": schema_distance_color,
                # "schema_distance_pos": schema_distance_pos,
                # "slots_distance": slots_distance,
            }


class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj
