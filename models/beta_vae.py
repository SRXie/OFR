import torch
import math
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple
from utils import Tensor
from utils import conv_transpose_out_shape
from utils import assert_shape
from utils import build_grid


class BetaVAE(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 128,
                 hidden_dims: List = None,
                 decoder_type: str = 'deconv',
                 beta: int = 4,
                 gamma:float = 10.,
                 decoder_resolution: Tuple[int, int] = (8, 8),
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.decoder_type = decoder_type
        self.beta = beta
        self.gamma = gamma
        self.decoder_resolution = decoder_resolution

        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        if decoder_type == 'deconv':
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )



            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                hidden_dims[-1],
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                output_padding=1),
                                nn.BatchNorm2d(hidden_dims[-1]),
                                nn.LeakyReLU(),
                                nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                        kernel_size= 3, padding= 1),
                                nn.Tanh())
        elif decoder_type == 'sbd':
            hidden_dims = [64, 64, 64]
            self.out_features = hidden_dims[-1]
            modules = []

            in_size = self.decoder_resolution[0]
            out_size = in_size

            for i in range(len(hidden_dims) - 1, -1, -1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            hidden_dims[i],
                            hidden_dims[i - 1],
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
                (64, 64),
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
                    nn.ConvTranspose2d(self.out_features, 3, kernel_size=3, stride=1, padding=1, output_padding=0,),
                )
            )

            assert_shape((64, 64), (out_size, out_size), message="")

            self.decoder = nn.Sequential(*modules)
            self.decoder_pos_embedding = SoftPositionEmbed(3, self.out_features, self.decoder_resolution)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        if self.decoder_type == 'deconv':
            result = self.decoder_input(z)
            result = result.view(-1, 512, 2, 2)
            result = self.decoder(result)
            result = self.final_layer(result)
        elif self.decoder_type == 'sbd':
            decoder_in = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])
            result = self.decoder_pos_embedding(decoder_in)
            result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class BetaTCVAE(nn.Module):
    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 128,
                 hidden_dims: List = None,
                 decoder_type: str = 'deconv',
                 anneal_steps: int = 200,
                 alpha: float = 1.,
                 beta: float =  6.,
                 gamma: float = 1.,
                 decoder_resolution: Tuple[int, int] = (8, 8),
                 **kwargs) -> None:
        super(BetaTCVAE, self).__init__()

        self.latent_dim = latent_dim
        self.anneal_steps = anneal_steps

        self.decoder_type = decoder_type

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]

        self.decoder_resolution = decoder_resolution
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 5, stride= 2, padding  = 2),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc = nn.Linear(hidden_dims[-1]*64, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

        # Build Decoder

        modules = []

        self.out_features = hidden_dims[-1]

        if self.decoder_type == 'deconv':
            self.decoder_input = nn.Linear(latent_dim, self.out_features*self.decoder_resolution[0]*self.decoder_resolution[1])
        else:
            self.decoder_input = nn.Linear(latent_dim, self.out_features)

        if True:
            in_size = self.decoder_resolution[0]
            out_size = in_size

            for i in range(len(hidden_dims) - 1, -1, -1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            hidden_dims[i],
                            hidden_dims[i - 1],
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
                (128, 128),
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
                    nn.ConvTranspose2d(self.out_features, 3, kernel_size=3, stride=1, padding=1, output_padding=0,),
                )
            )

            self.decoder = nn.Sequential(*modules)
            self.decoder_pos_embedding = SoftPositionEmbed(3, self.out_features, self.decoder_resolution)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.fc(result)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        if self.decoder_type == 'deconv':
            result = result.view(-1, self.out_features, self.decoder_resolution[0], self.decoder_resolution[1])
            result = self.decoder(result)
        elif self.decoder_type == 'sbd':
            result = result.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])
            result = self.decoder_pos_embedding(result)
            result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        # mu = mu.detach()
        # log_var = log_var.detach()
        z = self.reparameterize(mu, log_var)#.detach()
        return  [self.decode(z), input, mu, log_var, z]

    def log_density_gaussian(self, x: Tensor, mu: Tensor, logvar: Tensor):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]

        weight = 1 #kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input, reduction='sum')

        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim = 1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim = 1)

        batch_size, latent_dim = z.shape
        mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                                mu.view(1, batch_size, latent_dim),
                                                log_var.view(1, batch_size, latent_dim))

        # Reference
        # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
        dataset_size = (1 / kwargs['M_N']) * batch_size # dataset size
        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size -1)).to(input.device)
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss  = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.

        loss = recons_loss/batch_size + \
               self.alpha * mi_loss + \
               weight * (self.beta * tc_loss +
                         anneal_rate * self.gamma * kld_loss)

        return {'loss': loss,
                'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss,
                'TC_Loss':tc_loss,
                'MI_Loss':mi_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj
