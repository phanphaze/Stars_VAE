# file where the model architecture is defined

import torch
import torch.nn as nn
from src.config import latent_dimension_size
from torchvision import datasets 
from tqdm import tqdm 
from torchvision import transforms 

class VAE(nn.Module): #Variational Autoencoder
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 200, z_dim: int = 20): 
        super(VAE, self).__init__()

        # Encoder
        self.encoder_class = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.sigma = nn.Linear(hidden_dim, z_dim)
        
        # Decoder
        self.decoder_class = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, x):
        a = self.encoder_class(x)
        mu = self.mu(a)
        sigma = self.sigma(a)
        return mu, sigma

    def decode(self, z):
        return self.decoder_class(z)
        
    def forward(self, x):
        mu, sigma = self.encode(x)
        sigma = torch.exp(0.5 * sigma)  # Convert log variance to standard deviation
        z_reparameterized = mu + sigma*torch.randn_like(sigma)
        x_reconstructed = self.decode(z_reparameterized)
        return x_reconstructed, mu, sigma

# class CAE(nn.Module): #Conditional Autoencoder
#     def __init__(
#         self,
#         input_dim: int,
#         condition_dim: int,
#         latent_dim: int = latent_dimension_size,
#     ):
#         super(CAE, self).__init__()

#         self.input_dim = input_dim
#         self.condition_dim = condition_dim
#         self.latent_dim = latent_dim

# 		# input + condition -> hidden 1 -> (mu, logvar) -> z -> z + condition -> hidden 2 -> output
#         hidden_dim = max(input_dim * 2, 256)
        
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim + condition_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.LeakyReLU(0.2),
            
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.LeakyReLU(0.2),
#         )

#         self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim + condition_dim, hidden_dim // 2),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.LeakyReLU(0.2),
            
#             nn.Linear(hidden_dim // 2, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.LeakyReLU(0.2),
            
#             nn.Linear(hidden_dim, input_dim),
#         )

#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         x_cond = torch.cat([x, c], dim=-1)
#         hidden = self.encoder(x_cond)
#         mu = self.fc_mu(hidden)
#         logvar = self.fc_logvar(hidden)
#         return mu, logvar

#     def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
#         z_cond = torch.cat([z, c], dim=-1)
#         return self.decoder(z_cond)

#     def forward(
#         self,
#         x: torch.Tensor,
#         c: torch.Tensor,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         mu, logvar = self.encode(x, c)
        # z = self.reparameterize(mu, logvar)
        # decoded = self.decode(z, c)
        # return decoded, mu, logvar