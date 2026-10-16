from typing import Sequence
import math

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from .abstract import Elasticity, Plasticity, Material

import torch
import torch.nn as nn
from omegaconf import DictConfig


class PACNeoHookeanElasticity(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()


        self.youngs_modulus_log = nn.Parameter(torch.tensor([cfg.E], dtype=torch.float32).log())
        self.poissons_ratio = nn.Parameter(torch.tensor([cfg.nu], dtype=torch.float32))



        self.damping_coeff = float(getattr(cfg, "damping_coeff", 0.0))

    def forward(self, F: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        将风格二的应力计算改写成风格一的 batch PyTorch 版本。

        Args:
            F: deformation gradient (B, 3, 3)  <-> 风格二: new_F
            grad_v: velocity gradient (B, 3, 3) <-> 风格二: self.C[p, s]
                    如果不提供，则只返回纯弹性 Neo-Hookean（无阻尼项）

        Returns:
            kirchhoff_stress: (B, 3, 3)
        """
        assert F.ndim == 3 and F.shape[-2:] == (3, 3), "F must be (B, 3, 3)"


        E = self.youngs_modulus_log.exp()
        nu = self.poissons_ratio

        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


        B = F @ F.transpose(1, 2)                                # (B, 3, 3) <-> new_F @ new_F^T
        J = torch.linalg.det(F).view(-1, 1, 1)                   # (B, 1, 1) <-> new_F.determinant()
        I = torch.eye(3, dtype=F.dtype, device=F.device).unsqueeze(0)  # (1, 3, 3)




        kirchhoff_stress = mu * (B - I) + lam * torch.log(J) * I


        epsilon = 0.5 * (C + C.transpose(1, 2))    # (B, 3, 3) <-> 0.5*(grad_v + grad_v^T)
        kirchhoff_stress = kirchhoff_stress + self.damping_coeff * epsilon * J

        return kirchhoff_stress


class PACNewtonianViscousFluidElasticity(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.mu = nn.Parameter(torch.Tensor([cfg.mu]))
        self.kappa = nn.Parameter(torch.Tensor([cfg.kappa]))

    def forward(self, F: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        风格二 viscous_fluid 的应力计算，转换为风格一批量 PyTorch 写法。

        Args:
            F: deformation-like tensor (B, 3, 3). 但这里只用 F[:,0,0] 作为 J
               <-> 风格二: new_F，且 J = new_F[0,0]
            C: velocity gradient (B, 3, 3)
               <-> 风格二: self.C[p, s]

        Returns:
            stress: (B, 3, 3)
        """
        assert F.ndim == 3 and F.shape[-2:] == (3, 3), "F must be (B, 3, 3)"
        assert C.shape == F.shape, "C must be the same shape as F (B, 3, 3)"


        # E = self.youngs_modulus_log.exp()
        # nu = self.poissons_ratio
        # mu = E / (2.0 * (1.0 + nu))
        # lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # kappa = (2.0 / 3.0) * mu + lam
        mu = self.mu  
        kappa = self.kappa
        lam = kappa - 2.0 / 3.0 * mu



        J = F[:, 0, 0].view(-1, 1, 1)  # (B,1,1)

        I = torch.eye(3, dtype=F.dtype, device=F.device).unsqueeze(0)  # (1,3,3)




        stress = kappa * I * (J - 1.0 / (J ** 6))



        #   grad_v = self.C[p, s]
        #   epsilon = 0.5*(grad_v + grad_v^T)
        #   stress += mu * epsilon * J
        epsilon = 0.5 * (C + C.transpose(1, 2))
        stress = stress + mu * epsilon * J

        return stress


class PACSigmaElasticity(Elasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))

        if cfg.random:
            self.log_E.data.mul_(0.8)

    def forward(self, F: Tensor, C: Tensor) -> Tensor:
        E = self.log_E.exp()
        nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        # warp svd
        U, sigma, Vh = self.svd(F)

        epsilon = sigma.log()
        trace = epsilon.sum(dim=1, keepdim=True)
        tau = 2 * mu * epsilon + la * trace

        stress = torch.matmul(torch.matmul(U, torch.diag_embed(tau)), self.transpose(U))

        return stress
    
class PACNonNewtonianSigmaElasticity(Elasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.mu = nn.Parameter(torch.Tensor([cfg.mu]))
        self.kappa = nn.Parameter(torch.Tensor([cfg.kappa]))

    def forward(self, F: Tensor, C: Tensor) -> Tensor:
        mu = self.mu
        kappa = self.kappa
        lam = kappa - 2.0 / 3.0 * mu

        # warp svd
        U, sigma, Vh = self.svd(F)

        epsilon = sigma.log()
        trace = epsilon.sum(dim=1, keepdim=True)
        tau = 2 * mu * epsilon + lam * trace

        stress = torch.matmul(torch.matmul(U, torch.diag_embed(tau)), self.transpose(U))

        return stress


class PACIdentityPlasticity(Plasticity):
    def forward(self, F: Tensor) -> Tensor:
        return F
    

class PACDruckerPragerPlasticity(Plasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))
        self.friction_angle = nn.Parameter(torch.Tensor([cfg.friction_angle]))
        # self.register_buffer('cohesion', torch.Tensor([cfg.cohesion]))
        self.cohesion = float(getattr(cfg, "cohesion", 0.0))

        if cfg.random:
            self.log_E.data.mul_(0.8)
            self.friction_angle.data.mul_(0.8)

    def forward(self, F: Tensor) -> Tensor:

        E = self.log_E.exp()
        nu = self.nu
        friction_angle = self.friction_angle
        sin_phi = torch.sin(torch.deg2rad(friction_angle))
        alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
        cohesion = self.cohesion

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        # warp svd
        U, sigma, Vh = self.svd(F)

        # prevent NaN
        thredhold = 0.05
        sigma = torch.clamp_min(sigma, thredhold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)

        expand_epsilon = torch.ones_like(epsilon) * cohesion

        shifted_trace = trace - cohesion * self.dim
        cond_yield = (shifted_trace < 0).view(-1, 1)

        delta_gamma = epsilon_hat_norm + (self.dim * la + 2 * mu) / (2 * mu) * shifted_trace * alpha
        compress_epsilon = epsilon - (torch.clamp_min(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat

        epsilon = torch.where(cond_yield, compress_epsilon, expand_epsilon)

        F = torch.matmul(torch.matmul(U, torch.diag_embed(epsilon.exp())), Vh)

        return F

class PACVonMisesPlasticity(Plasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)


        self.log_E = nn.Parameter(torch.tensor([cfg.E], dtype=torch.float32).log())
        self.register_buffer("nu", torch.tensor([cfg.nu], dtype=torch.float32))


        self.sigma_y = nn.Parameter(torch.tensor([cfg.sigma_y], dtype=torch.float32))

        # self.register_buffer("plastic_viscosity", torch.tensor([cfg.plastic_viscosity], dtype=torch.float32))
        # self.register_buffer("dt", torch.tensor([cfg.dt], dtype=torch.float32))

        self.plastic_viscosity = float(getattr(cfg, "plastic_viscosity", -1e6))
        self.dt = float(getattr(cfg, "dt", 1.0 / 24 / 200))

        if getattr(cfg, "random", False):
            self.log_E.data.mul_(0.8)
            self.sigma_y.data.mul_(0.8)

    def forward(self, F: Tensor) -> Tensor:
        """
        F: [B, dim, dim]
        返回: [B, dim, dim]
        """
        E = self.log_E.exp()          # [1]
        nu = self.nu                  # [1]
        mu = E / (2.0 * (1.0 + nu))

        # warp svd
        U, sig, Vh = self.svd(F)      # sig: [B, dim]


        threshold = 0.05
        sig = torch.clamp_min(sig, threshold)


        b_trial = sig ** 2                               # [B, dim]   (b_trial = sig ** 2)
        epsilon = torch.log(sig)                         # [B, dim]   (epsilon = log(sig))
        trace_eps = epsilon.sum(dim=1, keepdim=True)     # [B, 1]     (trace_epsilon = epsilon.sum())

        epsilon_hat = epsilon - trace_eps / self.dim     # [B, dim]   (epsilon_hat = epsilon - mean)
        s_trial = 2.0 * mu * epsilon_hat                 # [B, dim]   (s_trial = 2 * mu[p] * epsilon_hat)

        s_trial_norm = torch.linalg.norm(s_trial, dim=1, keepdim=True)  # [B,1] (norm(s_trial))

        # y = ||s_trial|| - sqrt(2/3) * yield_stress
        y = s_trial_norm - math.sqrt(2.0 / 3.0) * self.sigma_y     # [B,1]


        cond_yield = (y > 0)  # [B,1]

        # mu_hat = mu * b_trial.sum()/dim
        mu_hat = mu * (b_trial.sum(dim=1, keepdim=True) / self.dim)     # [B,1]

        # s_new_norm = s_trial_norm - y / (1 + eta/(2*mu_hat*dt))
        eta = self.plastic_viscosity    # [1]
        dt = self.dt                    # [1]
        denom = 1.0 + eta / (2.0 * mu_hat * dt)                          # [B,1]
        s_new_norm = s_trial_norm - y / denom                             # [B,1]

        # s_new = (s_new_norm / s_trial_norm) * s_trial

        eps = 1e-12
        ratio = s_new_norm / (s_trial_norm + eps)                         # [B,1]
        s_new = ratio * s_trial                                            # [B,dim]


        H = s_new / (2.0 * mu) + trace_eps / self.dim                      # [B,dim]

        sig_yield = torch.exp(H)                                            # [B,dim]
        sig_out = torch.where(cond_yield, sig_yield, sig)                   # [B,dim] (else: sig_out = sig)


        F_out = U @ torch.diag_embed(sig_out) @ Vh                          # [B,dim,dim]
        return F_out

# class PACVonMisesNonNewtonianPlasticity(Plasticity):
#     def __init__(self, cfg: DictConfig) -> None:
#         super().__init__(cfg)


#         self.mu = nn.Parameter(torch.tensor([cfg.mu], dtype=torch.float32))
#         self.register_buffer("kappa", torch.tensor([cfg.kappa], dtype=torch.float32))


#         self.sigma_y = nn.Parameter(torch.tensor([cfg.sigma_y], dtype=torch.float32))

#         # self.register_buffer("plastic_viscosity", torch.tensor([cfg.plastic_viscosity], dtype=torch.float32))
#         # self.register_buffer("dt", torch.tensor([cfg.dt], dtype=torch.float32))

#         self.plastic_viscosity = float(getattr(cfg, "plastic_viscosity", -1e6))
#         self.dt = float(getattr(cfg, "dt", 1.0 / 24 / 200))
        

#     def forward(self, F: Tensor) -> Tensor:
#         """
#         F: [B, dim, dim]

#         """

#         kappa = self.kappa
#         lam = kappa - 2.0 / 3.0 * mu

#         # warp svd
#         U, sig, Vh = self.svd(F)      # sig: [B, dim]


#         threshold = 0.05
#         sig = torch.clamp_min(sig, threshold)


#         b_trial = sig ** 2                               # [B, dim]   (b_trial = sig ** 2)
#         epsilon = torch.log(sig)                         # [B, dim]   (epsilon = log(sig))
#         trace_eps = epsilon.sum(dim=1, keepdim=True)     # [B, 1]     (trace_epsilon = epsilon.sum())

#         epsilon_hat = epsilon - trace_eps / self.dim     # [B, dim]   (epsilon_hat = epsilon - mean)
#         s_trial = 2.0 * mu * epsilon_hat                 # [B, dim]   (s_trial = 2 * mu[p] * epsilon_hat)

#         s_trial_norm = torch.linalg.norm(s_trial, dim=1, keepdim=True)  # [B,1] (norm(s_trial))

#         # y = ||s_trial|| - sqrt(2/3) * yield_stress
#         y = s_trial_norm - math.sqrt(2.0 / 3.0) * self.sigma_y     # [B,1]


#         cond_yield = (y > 0)  # [B,1]

#         # mu_hat = mu * b_trial.sum()/dim
#         mu_hat = mu * (b_trial.sum(dim=1, keepdim=True) / self.dim)     # [B,1]

#         # s_new_norm = s_trial_norm - y / (1 + eta/(2*mu_hat*dt))
#         eta = self.plastic_viscosity    # [1]
#         dt = self.dt                    # [1]
#         denom = 1.0 + eta / (2.0 * mu_hat * dt)                          # [B,1]
#         s_new_norm = s_trial_norm - y / denom                             # [B,1]

#         # s_new = (s_new_norm / s_trial_norm) * s_trial

#         eps = 1e-12
#         ratio = s_new_norm / (s_trial_norm + eps)                         # [B,1]
#         s_new = ratio * s_trial                                            # [B,dim]


#         H = s_new / (2.0 * mu) + trace_eps / self.dim                      # [B,dim]

#         sig_yield = torch.exp(H)                                            # [B,dim]
#         sig_out = torch.where(cond_yield, sig_yield, sig)                   # [B,dim] (else: sig_out = sig)


#         F_out = U @ torch.diag_embed(sig_out) @ Vh                          # [B,dim,dim]
#         return F_out

class PACNonNewtonianVonMisesPlasticity(Plasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)


        self.mu = nn.Parameter(torch.tensor([cfg.mu], dtype=torch.float32))
        self.register_buffer("kappa", torch.tensor([cfg.kappa], dtype=torch.float32))


        self.sigma_y = nn.Parameter(torch.tensor([cfg.sigma_y], dtype=torch.float32))

        # self.register_buffer("plastic_viscosity", torch.tensor([cfg.plastic_viscosity], dtype=torch.float32))
        # self.register_buffer("dt", torch.tensor([cfg.dt], dtype=torch.float32))

        self.plastic_viscosity = float(getattr(cfg, "plastic_viscosity", -1e6))
        self.dt = float(getattr(cfg, "dt", 1.0 / 24 / 200))

    def forward(self, F: Tensor) -> Tensor:
        """
        输入:
        F: [B, dim, dim]
        输出:
        F_out: [B, dim, dim]
        """
        mu = self.mu
        kappa = self.kappa
        lam = kappa - 2.0 / 3.0 * mu                        # shear modulus

        # la = E * nu / ((1 + nu) * (1 - 2 * nu))


        U, sig, Vh = self.svd(F)                             # sig: [B, dim]


        threshold = 0.05
        sig = torch.clamp_min(sig, threshold)


        eps = torch.log(sig)                                  # [B, dim]
        trace_eps = eps.sum(dim=1, keepdim=True)              # [B, 1]
        eps_hat = eps - trace_eps / self.dim                  # [B, dim]



        s_trial = 2.0 * mu * eps_hat                           # [B, dim]
        s_trial_norm = torch.linalg.norm(s_trial, dim=1, keepdim=True)  # [B, 1]


        y = s_trial_norm - math.sqrt(2.0 / 3.0) * self.sigma_y     # [B, 1]
        yielded = (y > 0.0)                                             # [B, 1] bool


        # b_trial = sig**2; mu_hat = mu * b_trial.sum() / dim
        b_trial = sig * sig                                      # [B, dim]
        mu_hat = mu * (b_trial.sum(dim=1, keepdim=True) / self.dim)  # [B, 1]

        # s_new_norm = s_trial_norm - y / (1 + eta/(2*mu_hat*dt))
        denom = 1.0 + self.plastic_viscosity / (2.0 * mu_hat * self.dt)
        s_new_norm = s_trial_norm - (y / denom)


        s_new_norm = torch.clamp_min(s_new_norm, 0.0)

        # s_new = (s_new_norm / s_trial_norm) * s_trial

        safe_norm = torch.clamp_min(s_trial_norm, 1e-12)
        scale = s_new_norm / safe_norm
        s_new = scale * s_trial                                  # [B, dim]

        # H = s_new/(2*mu) + trace_eps/dim
        H = s_new / (2.0 * mu) + trace_eps / self.dim             # [B, dim]
        sig_plastic = torch.exp(H)                                # [B, dim]


        sig_elastic = sig                                         # [B, dim]


        sig_out = torch.where(yielded, sig_plastic, sig_elastic)  # [B, dim]


        F_out = U @ torch.diag_embed(sig_out) @ Vh                # [B, dim, dim]
        return F_out

# class PACViscousFluidPlasticity(Plasticity):
#     def __init__(self, cfg: DictConfig) -> None:
#         super().__init__(cfg)


#     def forward(self, F: Tensor) -> Tensor:
#         """
#         F: [B, dim, dim]
#         """
#         F_out = F.clone()
#         F_out[:, 0, 0] = torch.clamp_min(F_out[:, 0, 0], 0.05)
#         F = F_out
#         return F

class PACNewtonianSigmaPlasticity(Plasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def forward(self, F: Tensor) -> Tensor:
        J = torch.det(F)

        # unilateral incompressibility: https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Physics/PlasticityApplier.cpp#L1084
        # J = torch.clamp(J, min=0.05, max=1.2)

        Je_1_3 = torch.pow(J, 1.0 / 3.0).view(-1, 1).expand(-1, 3)
        F = torch.diag_embed(Je_1_3)
        return F