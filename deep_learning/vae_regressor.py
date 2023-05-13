import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import pyro
import pyro.distributions as dist
from deep_learning.decoder import Decoder

from deep_learning.encoder import Encoder


class VAE(nn.Module):
    def __init__(self, z_dim=20, hidden_dim=400, use_cuda=False):
        super().__init__()
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, x, y=None):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            loc_img = self.decoder(z)
            if y is not None:
                pyro.sample("obs", dist.Normal(loc_img, 1.0).to_event(1), obs=y)
            else:
                pyro.sample("obs", dist.Normal(loc_img, 1.0).to_event(1))

    def guide(self, x, y=None):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            print(len(x))
            z_loc, z_scale = self.encoder(x)
            #            z_loc, z_scale = self.encoder(x.reshape(-1, 1071))

            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        loc_img = self.decoder(z)
        return loc_img
