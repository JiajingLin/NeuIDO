from typing import Optional

from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange

from .abstract import Elasticity, Plasticity
from .utils import get_nonlinearity, get_norm, init_weight
from .meta import MLPBlock, MetaElasticity, MetaPlasticity
from .loralib import *

class ElasticityTrunckNetwork(MetaElasticity):
    def __init__(self, cfg: DictConfig, latent_dim: int) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()
        self.latent_dim = latent_dim
        
        width = self.dim + self.dim * self.dim + 1
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        # self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        self.final_layer = MLPBlock(width, latent_dim * (self.dim * self.dim), cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)
            
    def init_lora_layers(self, r: int, lora_alpha: int = 1):
        replace_with_linear_lora(self, nn.Linear, r, lora_alpha)
        print(f"Initialized LoRA layers in {type(self).__name__} with r={r} and alpha={lora_alpha}.")
    
    def freeze_all_except_lora(self):
        mark_only_lora_as_trainable(self)
    
    def lora_state_dict(self, bias: str = 'none'):
        return lora_state_dict(self, bias)

    def forward(self, F: Tensor, latents, traj_ids) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        U, sigma, Vh = self.svd(F)
        R = torch.matmul(U, Vh)

        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        if self.normalize_input:
            I1 = sigma - 1.0  # shape (batch_size, 3)
            I2 = self.flatten(FtF - I) # shape (batch_size, 9)
            I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0 # shape (batch_size, 1)
        else:
            I1 = sigma
            I2 = self.flatten(FtF)
            I3 = torch.linalg.det(F).unsqueeze(dim=1)

        x = torch.cat([I1, I2, I3], dim=1)
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_layer(x) # shape (batch_size, latent_dim * (self.dim * self.dim))

        x = x.reshape(x.shape[0], self.latent_dim, self.dim * self.dim) # shape (batch_size, latent_dim, self.dim * self.dim)
    

        if isinstance(latents, torch.nn.Embedding):
            latent_pts = latents(traj_ids)
        elif isinstance(latents, torch.Tensor):
            latent_pts = latents[traj_ids]
        else:
            raise TypeError(f"latents must be either torch.nn.Embedding or torch.Tensor, got {type(latents)}")
        # latent_pts = latents(traj_ids)  # shape (batch_size, latent_dim)


        x = torch.matmul(latent_pts.unsqueeze(dim=1), x) #  (batch_size, 1, latent_dim) * (batch_size, latent_dim, self.dim * self.dim) = (batch_size, 1, self.dim * self.dim)
        x = x.squeeze(dim=1) # shape (batch_size, self.dim * self.dim)
        x = self.unflatten(x) # shape (batch_size, self.dim, self.dim)
        
        x = 0.5 * (self.transpose(x) + x) 
        P = torch.matmul(R, x)
        cauchy = torch.matmul(P, Ft)
        return cauchy
    
class PlasticityTrunckNetwork(MetaPlasticity):
    def __init__(self, cfg: DictConfig, latent_dim: int) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()
        self.latent_dim = latent_dim

        width = 3 + 9 + 1
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, latent_dim * (self.dim * self.dim), cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)
            
    def init_lora_layers(self, r: int, lora_alpha: int = 1):
        replace_with_linear_lora(self, nn.Linear, r, lora_alpha)
        print(f"Initialized LoRA layers in {type(self).__name__} with r={r} and alpha={lora_alpha}.")
    
    def freeze_all_except_lora(self):
        mark_only_lora_as_trainable(self)
    
    def lora_state_dict(self, bias: str = 'none'):
        return lora_state_dict(self, bias)

    def forward(self, F: Tensor, latents, traj_ids) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        U, sigma, Vh = self.svd(F)
        R = torch.matmul(U, Vh)

        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        I1 = sigma - 1.0
        I2 = self.flatten(FtF - I)
        I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0
        
        invariants = torch.cat([I1, I2, I3], dim=1)
        x = invariants
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x) # shape (batch_size, latent_dim * (self.dim * self.dim))
        x = x.reshape(x.shape[0], self.latent_dim, self.dim * self.dim) # shape (batch_size, latent_dim, self.dim * self.dim)


        if isinstance(latents, torch.nn.Embedding):
            latent_pts = latents(traj_ids)
        elif isinstance(latents, torch.Tensor):
            latent_pts = latents[traj_ids]
        else:
            raise TypeError(f"latents must be either torch.nn.Embedding or torch.Tensor, got {type(latents)}")
        
        # latent_pts = latents(traj_ids)  # shape (batch_size, latent_dim)
        
        x = torch.matmul(latent_pts.unsqueeze(dim=1), x) #  (batch_size, 1, latent_dim) * (batch_size, latent_dim, self.dim * self.dim) = (batch_size, 1, self.dim * self.dim)
        x = x.squeeze(dim=1) # shape (batch_size, self.dim * self.dim)
        x = self.unflatten(x) # shape (batch_size, self.dim, self.dim)
        
        x = 0.5 * (self.transpose(x) + x)
        delta_Fp = self.alpha * torch.matmul(R, x)
        Fp = delta_Fp + F
        return Fp
