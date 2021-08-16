from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from slot_attention.utils import Tensor
from slot_attention.utils import assert_shape
from slot_attention.utils import build_grid
from slot_attention.utils import conv_transpose_out_shape
from slot_attention.utils import compute_cos_distance
from slot_attention.utils import batched_index_select


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
        hidden_dims: Tuple[int, ...] = (64, 64, 64), #delete one entry for 128 -> 64
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

    def forward(self, x, slots_only=False, dup_threshold=None):
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

        # to keep the one with largest attention mass in dup removal, sort slots by attention mass
        attn_mass = attn.permute(0,2,1).sum(-1)
        idx = torch.argsort(attn_mass, dim=1, descending=True)
        slots = batched_index_select(slots, 1, idx)
        attn = batched_index_select(attn, 2, idx)

        slots_nodup = slots.clone()
        if dup_threshold:
            cos_dis_pixel = compute_cos_distance(attn.permute(0,2,1)) # to have shape (batch_size, num_slot, emb_size)
            cos_dis_feature = compute_cos_distance(slots)
            cos_dis_min = cos_dis_feature # torch.min(cos_dis_pixel, cos_dis_feature)
            duplicated = cos_dis_min < dup_threshold
            # we only need the upper triangle
            duplicated = torch.triu(duplicated, diagonal=1)
            duplicated = torch.sum(duplicated, dim=1)
            duplicated_index = torch.nonzero(duplicated, as_tuple=True)
            # sample blank slots
            slots_init = torch.randn((duplicated_index[0].shape[0], slot_size))
            slots_init = slots_init.type_as(slots)
            self.slots_mu = self.slots_mu.to(slots.device)
            self.slots_log_sigma = self.slots_log_sigma.to(slots.device)
            blank_slots = self.slots_mu.squeeze(0) # + self.slots_log_sigma.squeeze(0).exp() * slots_init
            # fill in deuplicated slots with blank slots
            slots_nodup[duplicated_index[0], duplicated_index[1]] = blank_slots

        if slots_only:
            return slots, attn, slots_nodup

        slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
        if dup_threshold:
            slots_nodup = slots_nodup.view(batch_size * num_slots, slot_size, 1, 1)
            slots_cat = torch.cat([slots, slots_nodup])
            batch_size = batch_size*2
        else:
            slots_cat = slots
        decoder_in = slots_cat.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        assert_shape(out.size(), (batch_size * num_slots, num_channels + 1, height, width))

        out = out.view(batch_size, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)

        # masks_sum = masks.view(batch_size*num_slots, height*width).sum(-1)
        # blank_masks = masks_sum < height*width*0.001
        # index = torch.nonzero(blank_masks).squeeze(1)
        # if not torch.is_tensor(self.blank_slot):
        #     self.blank_slot = torch.rand_like(self.slots_mu.squeeze(0).squeeze(0))
        # if not index.shape[0] == 0:
        #     blank_slots = slots.view(batch_size*num_slots, -1)[index].mean(0)
        #     self.blank_slot = 0.995 * self.blank_slot + 0.005*blank_slots

        recon_combined = torch.sum(recons * masks, dim=1)

        if dup_threshold:
            batch_size = batch_size//2
            recons, recons_nodup = torch.split(recons, batch_size, 0)
            masks, masks_nodup = torch.split(masks, batch_size, 0)
            recon_combined, recon_combined_nodup = torch.split(recon_combined, batch_size, 0)
            slots_nodup = slots_nodup.view(batch_size, num_slots, slot_size)
        slots = slots.view(batch_size, num_slots, slot_size)
        if dup_threshold:
            return recon_combined, recons, masks, slots, attn, recon_combined_nodup, recons_nodup, masks_nodup, slots_nodup
        else:
            return recon_combined, recons, masks, slots, attn

    def loss_function(self, input):
        recon_combined, recons, masks, slots, attn = self.forward(input)
        loss = F.mse_loss(recon_combined, input)
        return {
            "loss": loss,
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
