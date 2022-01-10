import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import init_weights, _softplus_to_std, mvn, std_mvn
from utils import gmm_loglikelihood
from utils import compute_cos_distance
from utils import batched_index_select
from utils import compute_mask_ari
from utils import compute_corr_coef
from utils import assert_shape
from utils import compute_greedy_loss, compute_pseudo_greedy_loss
import numpy as np

class RefinementNetwork(nn.Module):
    def __init__(self, z_size, input_size, refinenet_channels_in=16, conv_channels=64, lstm_dim=256):
        super(RefinementNetwork, self).__init__()
        self.input_size = input_size
        self.z_size = z_size

        self.conv = nn.Sequential(
            nn.Conv2d(refinenet_channels_in, conv_channels, 3, 2, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 3, 2, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 3, 2, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 3, 2, 1),
            nn.ELU(True),
            nn.AvgPool2d(4),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear((input_size[1]//64)*(input_size[1]//64)*conv_channels, lstm_dim),
            nn.ELU(True)
        )

        self.input_proj = nn.Sequential(
                nn.Linear(lstm_dim + 4*self.z_size, lstm_dim),
                nn.ELU(True)
            )
        self.lstm = nn.LSTM(lstm_dim, lstm_dim)
        self.loc = nn.Linear(lstm_dim, z_size)
        self.softplus = nn.Linear(lstm_dim, z_size)

    def forward(self, img_inputs, vec_inputs, h, c):
        """
        img_inputs: [N * K, C, H, W]
        vec_inputs: [N * K, 4*z_size]
        """
        x = self.conv(img_inputs)
        x = self.mlp(x)
        # concat with \lambda and \nabla \lambda
        x = torch.cat([x, vec_inputs], 1)
        x = self.input_proj(x)
        x = x.unsqueeze(0) # seq dim
        self.lstm.flatten_parameters()
        out, (h,c) = self.lstm(x, (h,c))
        out = out.squeeze(0)
        loc = self.loc(out)
        softplus = self.softplus(out)
        lamda = torch.cat([loc, softplus], 1)
        return lamda, (h,c)


class SpatialBroadcastDecoder(nn.Module):
    """
    Decodes the individual Gaussian image componenets
    into RGB and mask. This is the architecture used for the
    Multi-dSprites experiment but I haven't seen any issues
    with re-using it for CLEVR. In their paper they slightly
    modify it (e.g., uses 3x3 conv instead of 5x5).
    """
    def __init__(self, input_size, z_size, conv_channels=64):
        super(SpatialBroadcastDecoder, self).__init__()
        self.h, self.w = input_size[1], input_size[2]
        self.decode = nn.Sequential(
            nn.Conv2d(z_size+2, conv_channels, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, 4, 3, 1, 1)
        )


    @staticmethod
    def spatial_broadcast(z, h, w):
        """
        source: https://github.com/baudm/MONet-pytorch/blob/master/models/networks.py
        """
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, z):
        z_sb = SpatialBroadcastDecoder.spatial_broadcast(z, self.h, self.w)
        out = self.decode(z_sb) # [batch_size * K, output_size, h, w]
        return torch.sigmoid(out[:,:3]), out[:,3]


class IODINE(nn.Module):
    def __init__(self, z_size, resolution, num_slots, num_iters, log_scale=math.log(0.10), kl_beta=1, lstm_dim=256):
        super(IODINE, self).__init__()

        self.z_size = z_size
        self.input_size = [3]+list(resolution)
        self.K = num_slots
        self.inference_iters = num_iters
        self.kl_beta = kl_beta
        self.lstm_dim = lstm_dim
        self.gmm_log_scale = log_scale * torch.ones(self.K)
        self.gmm_log_scale = self.gmm_log_scale.view(1, self.K, 1, 1, 1)

        self.image_decoder = SpatialBroadcastDecoder(z_size=self.z_size, input_size=self.input_size)
        self.refine_net = RefinementNetwork(z_size=self.z_size, input_size=self.input_size) # 16 is the concatnation of all refine inputs

        init_weights(self.image_decoder, 'xavier')
        init_weights(self.refine_net, 'xavier')

        # learnable initial posterior distribution
        # loc = 0, variance = 1
        self.lamda_0 = nn.Parameter(torch.cat([torch.zeros(1,self.z_size),torch.ones(1,self.z_size)],1))

        # layernorms for iterative inference input
        n = self.input_size[1]
        self.layer_norms = torch.nn.ModuleList([
                nn.LayerNorm((1,n,n), elementwise_affine=False),
                nn.LayerNorm((1,n,n), elementwise_affine=False),
                nn.LayerNorm((3,n,n), elementwise_affine=False),
                nn.LayerNorm((1,n,n), elementwise_affine=False),
                nn.LayerNorm((self.z_size,), elementwise_affine=False), # layer_norm_mean
                nn.LayerNorm((self.z_size,), elementwise_affine=False)  # layer_norm_log_scale
            ])


    @staticmethod
    @torch.enable_grad()
    def refinenet_inputs(image, means, masks, mask_logits, log_p_k, normal_ll, lamda, loss, layer_norms, eval_mode):
        N, K, C, H, W = image.shape
        # non-gradient inputs
        # 1. image [N, K, C, H, W]
        # 2. means [N, K, C, H, W]
        # 3. masks  [N, K, 1, H, W] (log probs)
        # 4. mask logits [N, K, 1, H, W]
        # 5. mask posterior [N, K, 1, H, W]
        normal_ll = torch.sum(normal_ll, dim=2)
        mask_posterior = (normal_ll - torch.logsumexp(normal_ll, dim=1).unsqueeze(1)).unsqueeze(2) # logscale
        # 6. pixelwise likelihood [N, K, 1, H, W]
        log_p_k = torch.logsumexp(log_p_k, dim=[1,2])
        log_p_k = log_p_k.view(-1, 1, 1, H, W).repeat(1, K, 1, 1, 1)
        px_l = log_p_k  # log scale
        #px_l = log_p_k.exp() # not log scale
        # 7. LOO likelihood
        #loo_px_l = torch.log(1e-6 + (px_l.exp()+1e-6 - (masks + normal_ll.unsqueeze(2).exp())+1e-6)) # [N,K,1,H,W]

        # 8. Coordinate channels
        x = torch.linspace(-1, 1, W, device='cuda')
        y = torch.linspace(-1, 1, H, device='cuda')
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, k, 1, h, w)
        x_mesh = x_b.expand(N, K, 1, -1, -1)
        y_mesh = y_b.expand(N, K, 1, -1, -1)

        # 9. \partial L / \partial means
        # [N, K, C, H, W]
        # 10. \partial L/ \partial masks
        # [N, K, 1, H, W]
        # 11. \partial L/ \partial lamda
        # [N*K, 2 * self.z_size]
        d_means, d_masks, d_lamda = \
                torch.autograd.grad(loss, [means, masks, lamda], create_graph=not eval_mode,
                        retain_graph=not eval_mode, only_inputs=True)

        d_loc_z, d_sp_z = d_lamda.chunk(2, dim=1)
        d_loc_z, d_sp_z = d_loc_z.contiguous(), d_sp_z.contiguous()

        # apply layernorms
        px_l = layer_norms[0](px_l).detach()
        #loo_px_l = layer_norms[1](loo_px_l).detach()
        d_means = layer_norms[2](d_means).detach()
        d_masks = layer_norms[3](d_masks).detach()
        d_loc_z = layer_norms[4](d_loc_z).detach()
        d_sp_z = layer_norms[5](d_sp_z).detach()

        # concat image-size and vector inputs
        image_inputs = torch.cat([
            image, means, masks, mask_logits, mask_posterior, px_l,
            d_means, d_masks, x_mesh, y_mesh], 2)
        vec_inputs = torch.cat([
            lamda, d_loc_z, d_sp_z], 1)
        return image_inputs.view(N * K, -1, H, W), vec_inputs

    @torch.enable_grad()
    def forward(self, x, training=False, dup_threshold=None, viz=False):
        """
        Evaluates the model as a whole, encodes and decodes
        and runs inference for T steps
        """
        C, H, W = self.input_size[0], self.input_size[1], self.input_size[2]
        batch_size  = x.shape[0]
        num_slots = self.K
        slot_size = self.z_size
        # expand lambda_0
        lamda = self.lamda_0.repeat(batch_size*self.K,1) # [N*K, 2*z_size]
        p_z = std_mvn(shape=[batch_size * self.K, self.z_size], device=x.device)

        total_loss = 0.
        losses = []
        x_means = []
        mask_logps = []
        h, c = (torch.zeros(1, batch_size*self.K, self.lstm_dim),
                    torch.zeros(1, batch_size*self.K, self.lstm_dim))
        h = h.to(x.device)
        c = c.to(x.device)

        for i in range(self.inference_iters):
            # sample initial posterior
            loc_z, sp_z = lamda.chunk(2, dim=1)
            loc_z, sp_z = loc_z.contiguous(), sp_z.contiguous()
            q_z = mvn(loc_z, sp_z)
            z = q_z.rsample()

            # Get means and masks
            x_loc, mask_logits = self.image_decoder(z)  #[N*K, C, H, W]
            x_loc = x_loc.view(batch_size, self.K, C, H, W)

            # softmax across slots
            mask_logits = mask_logits.view(batch_size, self.K, 1, H, W)
            mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)

            # NLL [batch_size, 1, H, W]
            log_var = (2 * self.gmm_log_scale).to(x.device)
            nll, ll_outs = gmm_loglikelihood(x, x_loc, log_var, mask_logprobs)

            # KL div
            kl_div = torch.distributions.kl.kl_divergence(q_z, p_z)
            kl_div = kl_div.view(batch_size, self.K).sum(1)
            loss = nll + self.kl_beta * kl_div
            loss = torch.mean(loss)
            scaled_loss = ((i+1.)/self.inference_iters) * loss
            losses += [scaled_loss]
            total_loss += scaled_loss

            x_means += [x_loc]
            mask_logps += [mask_logprobs]

            # Refinement
            if i == self.inference_iters-1:
                # after T refinement steps, just output final loss
                recons = x_loc
                masks = nn.functional.softmax(mask_logits, dim=1)
                recon_combined = torch.sum(masks * recons, dim=1)
                slots = z.view(batch_size, num_slots, slot_size)

                if dup_threshold:
                    slots_nodup = slots.clone()
                    cos_dis_feature = compute_cos_distance(slots)
                    duplicated = cos_dis_feature < dup_threshold
                    # we only need the upper triangle
                    duplicated = torch.triu(duplicated, diagonal=1)
                    duplicated = torch.sum(duplicated, dim=1)
                    duplicated_index = torch.nonzero(duplicated, as_tuple=True)
                    # get blank slots
                    blank_slots = p_z.mean[0]
                    # fill in deuplicated slots with blank slots
                    slots_nodup[duplicated_index[0], duplicated_index[1]] = blank_slots

                    masks_mass = masks.view(batch_size, num_slots, -1)
                    masks_mass = torch.where(masks_mass>=masks_mass.max(dim=1)[0].unsqueeze(1).repeat(1,recons.shape[1],1), masks_mass, torch.zeros_like(masks_mass)).sum(-1)
                    invisible_index = torch.nonzero(masks_mass==0.0, as_tuple=True)
                    slots_nodup[invisible_index[0], invisible_index[1]] = blank_slots

                    mask_logits[duplicated_index[0], duplicated_index[1]] = -1000000.0*torch.ones_like(masks_mass[0,0])
                    masks_nodup = F.softmax(mask_logits, dim=1)
                    recons_nodup = recons
                    recon_combined_nodup = torch.sum(recons * masks_nodup, dim=1)

                    if viz:
                        # Here we reconstruct D'
                        batch_size = batch_size//4
                        _, cat_indices = compute_greedy_loss(slots_nodup) #compute_pseudo_greedy_loss(slots, [])
                        slots_nodup = batched_index_select(slots_nodup, 1, cat_indices) # batched_index_select(slots, 1, cat_indices)
                        recons_nodup = batched_index_select(recons, 1, cat_indices)
                        masks_nodup = batched_index_select(masks_nodup, 1, cat_indices)
                        slots_D_prime=slots_nodup[:batch_size]-slots_nodup[batch_size:2*batch_size]+slots_nodup[2*batch_size:3*batch_size]
                        slots_D_prime = slots_D_prime.view(batch_size*num_slots, -1)
                        recons_D_prime, mask_logits_D_prime = self.image_decoder(slots_D_prime)
                        recons_D_prime=recons_D_prime.view(batch_size, self.K, C, H, W)
                        mask_logits_D_prime=mask_logits_D_prime.view(batch_size, self.K, 1, H, W)
                        slots_D_prime = slots_D_prime.view(batch_size, num_slots, slot_size)

                        # Here we mask the de-duplicated slots in generation
                        detect_blank_slots = slots_D_prime.clone()
                        detect_blank_slots = ((detect_blank_slots - blank_slots.unsqueeze(0)).sum(-1)==0.0).nonzero(as_tuple=True)
                        masks_D_prime = F.softmax(mask_logits_D_prime, dim=1)
                        masks_D_prime[detect_blank_slots[0], detect_blank_slots[1]] = -10000.0*torch.ones_like(masks_D_prime[0,0])
                        recon_combined_D_prime = torch.sum(recons_D_prime * masks_D_prime, dim=1)

                        recon_combined_nodup = torch.cat(list(recon_combined_nodup.split(batch_size, 0))+[recon_combined_D_prime], 0)
                        recons_nodup = torch.cat(list(recons_nodup.split(batch_size, 0))+[recons_D_prime], 0)
                        masks_nodup = torch.cat(list(masks_nodup.split(batch_size, 0))+[masks_D_prime], 0)
                        slots_nodup = torch.cat(list(slots_nodup.split(batch_size, 0))+[slots_D_prime], 0)

                    return recon_combined, recons, masks, slots, recon_combined_nodup, recons_nodup, masks_nodup, slots_nodup
                continue

            # compute refine inputs
            x_ = x.repeat(self.K, 1, 1, 1).view(batch_size, self.K, C, H, W)

            img_inps, vec_inps = IODINE.refinenet_inputs(x_, x_loc, mask_logprobs,
                    mask_logits, ll_outs['log_p_k'], ll_outs['normal_ll'], lamda, loss, self.layer_norms, not training)

            delta, (h,c) = self.refine_net(img_inps, vec_inps, h, c)
            lamda = lamda + delta


        return recon_combined, masks, recons, z, total_loss, torch.mean(nll), torch.mean(kl_div)

    def loss_function(self, x, mask_gt=None, schema_gt=None):
        """
        :param x: (B, 3, H, W)
        :return: loss
        """
        B, _, H, W = x.size()
        recon_combined, masks, recons, slots, total_loss, nll, kl = self.forward(x, training=True)
        if not mask_gt:
            return {
                "elbo": total_loss,
                "loss": nll,
                "kl": kl,
            }
        else:
            # compute ARI with mask gt
            # (batch_size, num_slots, 1, H, W) to (batch_size, num_slots, H, W)
            pred_mask = masks.squeeze(2)

            batch_size, num_slots, H, W = pred_mask.size()
            mask_gt = torch.stack(mask_gt, 1)[:,:,0,:,:]
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

            return {
                "elbo": total_loss,
                "loss": nll,
                "kl": kl,
                "mask_ari": mask_ari,
            }
