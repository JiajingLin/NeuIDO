import sys

import dataclasses
import random
from pathlib import Path
import argparse
import yaml

import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import warp as wp

from sim_modules.dataset import video_guidance
from sim_modules.sim import MPMModelBuilder, MPMStateInitializer, MPMStaticsInitializer, MPMInitData, MPMForwardSim, MPMDiffSim, MPMCacheDiffSim

from omegaconf import DictConfig, OmegaConf

from sim_modules.utils.config_helper import parse_yaml_config, parse_unknown_args
from d3gs.scene.gaussian_model import GaussianModel

import os

from sim_modules.utils.binding_helper import *
from d3gs.utils.render_utils import diff_rasterization
from sim_modules.dataset.video_guidance import evaluate
from sim_modules.utils.visualization_helper import save_video, save_render_video, save_render_gt_video

import copy

import sim_modules
from sim_modules.dataset.video_guidance import VideoGuidance
from sim_modules.config import VisionConfig
from d3gs.utils.loss_utils import l1_loss, l2_loss
from torchvision.utils import save_image
import omegaconf
from operator_modules.branch_modules.vae.utils import instantiate_from_config
from operator_modules.branch_modules.vae.autoencoder import VideoAutoencoderInferenceWrapper
from operator_modules.branch_modules.vae.cp_enc_dec import ContextParallelEncoder3D
from operator_modules.branch_modules.vae.context_parallel_util import initialize_context_parallel, set_context_parallel_group
from einops import rearrange, repeat
import mpu
from torchvision.utils import flow_to_image
from operator_modules.trunck_modules.trunck import PlasticityTrunckNetwork, ElasticityTrunckNetwork
from operator_modules.branch_modules.branch import VideoBranchEncoder
import logging

def check_tensor_finite(name, t, logger=None, raise_on_bad=True, max_abs_warn=None):
    with torch.no_grad():
        if not torch.isfinite(t).all():
            msg = f"[CHECK FAILED] {name} has NaN/Inf. " \
                  f"min={t.min().item():.4e}, max={t.max().item():.4e}"
            if logger is not None:
                logger.error(msg)
            else:
                print(msg)
            if raise_on_bad:
                raise RuntimeError(msg)
            return False

        max_abs = t.abs().max().item()
        if max_abs_warn is not None and max_abs > max_abs_warn:
            msg = f"[CHECK WARN] {name} abs.max={max_abs:.4e} > {max_abs_warn:.4e}"
            if logger is not None:
                logger.warning(msg)
            else:
                print(msg)
        return True


def check_F_det(name, F, logger=None, raise_on_bad=True):
    with torch.no_grad():
        J = torch.linalg.det(F)
        J_min = J.min().item()
        J_max = J.max().item()
        if not torch.isfinite(J).all() or torch.any(J <= 0.0):
            msg = f"[CHECK FAILED] det({name}) invalid. " \
                  f"min={J_min:.4e}, max={J_max:.4e}"
            if logger is not None:
                logger.error(msg)
            else:
                print(msg)
            if raise_on_bad:
                raise RuntimeError(msg)
            return False
        return True

@torch.no_grad()
def forward(config: str, scene_name: str, material_model_path: str):

    cfg = OmegaConf.load(config)
    cfg = DictConfig(cfg)

    if '_' in scene_name:

        scene_name, scene_id = scene_name.split('_')
        scene_cfg = os.path.join(cfg.scene_cfg_base, f"{scene_name}.yaml")
        scene_cfg = OmegaConf.load(scene_cfg)
        scene_cfg = DictConfig(scene_cfg)
        scene_cfg.gaussians.kernels_path = scene_cfg.gaussians.kernels_path.format(scene_id=scene_id)
        scene_cfg.particles.shape.asset_root = scene_cfg.particles.shape.asset_root.format(scene_id=scene_id)
        scene_cfg.dataset.data.path = scene_cfg.dataset.data.path.format(scene_id=scene_id)
        scene_cfg.dataset.data.init_path = scene_cfg.dataset.data.init_path.format(scene_id=scene_id)
    else:
        scene_cfg = os.path.join(cfg.scene_cfg_base, f"{scene_name}.yaml")
        scene_cfg = OmegaConf.load(scene_cfg)
        scene_cfg = DictConfig(scene_cfg)

    cfg = OmegaConf.merge(cfg, scene_cfg)
    
    wp.config.quiet = True
    wp.init()
    wp_device = wp.get_device(f'cuda:{cfg.gpu}')
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': True})

    torch_device = torch.device(f'cuda:{cfg.gpu}')
    torch.backends.cudnn.benchmark = True
    # sim_modules.warp.replace_torch_svd()
    # sim_modules.warp.replace_torch_polar()
    # sim_modules.warp.replace_torch_trace()
    # sim_modules.warp.replace_torch_cbrt()
    torch.set_default_device(torch_device)
    
    local_dir = cfg.local_dir
    save_name = 'evaluation_few_shot'
    save_name = f"{save_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_root = Path(f"{local_dir}/{save_name}")
    sim_modules.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
    sim_images_root = exp_root / 'images'
    sim_pts_root =  exp_root / 'pts'
    sim_images_root.mkdir(parents=True, exist_ok=True)
    sim_pts_root.mkdir(parents=True, exist_ok=True)
     
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(f"{local_dir}/{save_name}/logs", exist_ok=True)
    fh = logging.FileHandler(f"{local_dir}/{save_name}/log.txt")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    fh.setLevel(logging.INFO)
    # fh.setLevel(logging.DEBUG) # or any level you want
    logger.addHandler(fh)
     
    #################################################################
    # Define Constutitve Model Starting
    ################################################################# 
    latent_dim = cfg.operator.latent_dim
    material_model = torch.load(material_model_path, map_location=torch_device)

    if material_model['motion_elasticity_latent'] is not None:
        elasticity_latent_ckpt = material_model['motion_elasticity_latent']

        latent_mean = elasticity_latent_ckpt["weight"].mean(dim=0)
        motion_elasticity_latent = torch.nn.Embedding(1, latent_dim).to(torch_device) # shape (1, latent_dim)
        motion_elasticity_latent.weight.data = latent_mean.unsqueeze(0)
    else:
        motion_elasticity_latent = torch.nn.Embedding(1, latent_dim).cuda()
        nn.init.normal_(motion_elasticity_latent.weight, mean=0.0, std=1.0)
        print("No pretrained elasticity latent found, initializing from scratch.")
    
    if material_model['motion_plasticity_latent'] is not None:
        plasticity_latent_ckpt = material_model['motion_plasticity_latent']
        latent_mean = plasticity_latent_ckpt["weight"].mean(dim=0)
        motion_plasticity_latent = torch.nn.Embedding(1, latent_dim).to(torch_device) # shape (1, latent_dim)
        motion_plasticity_latent.weight.data = latent_mean.unsqueeze(0)
    else:
        motion_plasticity_latent = torch.nn.Embedding(1, latent_dim).cuda()
        nn.init.normal_(motion_plasticity_latent.weight, mean=0.0, std=1.0)
        print("No pretrained plasticity latent found, initializing from scratch.")
        
    ## Load the model ##
    elasticity: nn.Module = ElasticityTrunckNetwork(cfg.operator.trunck.elasticity, latent_dim)
    elasticity.to(torch_device)
    elasticity.eval()
    elasticity.requires_grad_(False)

    plasticity: nn.Module = PlasticityTrunckNetwork(cfg.operator.trunck.plasticity, latent_dim)
    plasticity.to(torch_device)
    plasticity.eval()
    plasticity.requires_grad_(False)        
    
    trunk_path = cfg.infer.pretrained_trunk
    if trunk_path is not None:
        ckpt = torch.load(trunk_path, map_location=torch_device)
        elasticity.load_state_dict(ckpt["elasticity"])
        plasticity.load_state_dict(ckpt["plasticity"])

        print(f"Loaded pretrained trunk from {trunk_path}")
    else:
        print("No pretrained trunk found, initializing from scratch.")
        
    if material_model['elasticity'] is not None and material_model['plasticity'] is not None:
        elasticity.init_lora_layers(r=cfg.operator.trunck.lora.r, lora_alpha=cfg.operator.trunck.lora.alpha)
        plasticity.init_lora_layers(r=cfg.operator.trunck.lora.r, lora_alpha=cfg.operator.trunck.lora.alpha)
        elasticity.load_state_dict(material_model['elasticity'], strict=False)
        plasticity.load_state_dict(material_model['plasticity'], strict=False)
        print(f"Loaded pretrained trunk lora from {material_model_path}")
    else:
        print("No pretrained trunk lora found, initializing from scratch.")
    
    #################################################################
    # Define Constutitve Model Ending
    #################################################################
    
    
    
    #################################################################
    # Define Gaussian Model Starting
    #################################################################   

    force_mask_data = cfg.dataset.data.read_mask_only
    if force_mask_data:
        # Force to use black background when loading mask data
        cfg.dataset.data.white_background = False
        print(f"[Warning] Force to use black background when loading mask data")

    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=torch_device)
        if cfg.dataset.data.white_background  # default to black background
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=torch_device)
    )

    # gaussians
    gaussians = GaussianModel(cfg.gaussians.sh_degree)
    gaussians.load_ply(cfg.gaussians.kernels_path, requires_grad=False)  # load Gaussian model from PLY file
    ################################################################
    # Define Gaussian Model Ending
    ################################################################

    #################################################################
    # Define Binding Data Starting
    #################################################################
    # -- binding data
    bind_data = torch.load(os.path.join(os.path.dirname(cfg.gaussians.kernels_path),'bindings.pt'))
    # bindings: torch.Tensor = bind_data['bindings'].to(torch_device).float()
    bindings: torch.Tensor = torch.sparse_coo_tensor(
        bind_data['bindings_ind'], bind_data['bindings_val'], bind_data['bindings_size']
    ).to(torch_device).float() 
    #################################################################
    # Define Binding Data Ending
    #################################################################
    
    #################################################################
    # Define Guidance Starting
    #################################################################
    cfg.dataset.data.skip_frames = 5
    # cfg.dataset.data.used_views = ['d_0', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5', 'd_6', 'd_7', 'd_8', 'd_9']
    cfg.dataset.data.used_views = ['d_0']
    # cfg.dataset.data.used_views = args.debug_views
    guidance = VideoGuidance(cfg)
    print("\n===================================")  
    print(f'Loading initial velocity from checkpoint ...\n')
    init_x_and_v = torch.load(cfg.dataset.data.init_path, map_location="cpu")
    guidance.set_init_x_and_v(init_x=init_x_and_v['init_x'], init_v=init_x_and_v['init_v'])
    print(f'\nInitial velocity obtained: {guidance.get_init_v.mean(0)}.')
    print("===================================")
    #################################################################
    # Define Guidance Ending
    #################################################################
    

    #################################################################
    # Define MPM Model Starting
    #################################################################
    # warp

    model = MPMModelBuilder().parse_cfg(cfg.physics.sim).finalize(wp_device)
    state_initializer = MPMStateInitializer(model)
    statics_initializer = MPMStaticsInitializer(model)

    init_data = MPMInitData.get(cfg.particles)

    init_data.set_ind_vel(guidance.get_init_v.cpu().numpy())

    state_initializer.add_group(init_data)
    statics_initializer.add_group(init_data)

    state, _ = state_initializer.finalize()
    statics = statics_initializer.finalize()
    # sim = MPMForwardSim(model, statics)    
    sim = MPMDiffSim(model, statics)
    #################################################################
    # Define MPM Model Ending
    #################################################################
    
    
    

    x, v, C, F, _ = guidance.get_init_material_data()
        
    de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)

    de_x_prev = de_x.clone().detach()
    g_prev = gaussians.get_xyz.clone().detach()
    
    render_dict = {}
    gt_dict = {}
    
    renders = []
    gts = []
    
    """注意此处"""
    num_steps = cfg.dataset.data.nframes * cfg.physics.sim.substeps
    for it in trange(0, num_steps, desc=f'[eval] {save_name}', file=sys.stdout, leave=None):

        traj_id = torch.zeros(F.shape[0], dtype=torch.long, device=torch_device)
        stress = elasticity(F, motion_elasticity_latent, traj_ids=traj_id)
        # state.from_torch(F=F, stress=stress)
        # x, v, C, F = sim(state)
        x, v, C, F = sim(x, v, C, F, stress)
        F = plasticity(F, motion_plasticity_latent, traj_ids=traj_id)

        if (it + 1) % (cfg.physics.sim.substeps * cfg.dataset.data.skip_frames) == 0:
        # if (it + 1) % cfg.dataset.data.skip_frames == 0:
            cur_frame = (it + 1) // (cfg.physics.sim.substeps * cfg.dataset.data.skip_frames)
            # cur_frame = (it + 1) // cfg.dataset.data.skip_frames
            de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)
            """在此处实现的particle-GS binding的计算"""
            means3D = compute_bindings_xyz(de_x, de_x_prev, g_prev, bindings)
            deform_grad = compute_bindings_F(F, bindings)
            """在此处实现的particle-GS binding的计算"""
            

            t = trimesh.PointCloud(vertices=de_x.clone().detach().cpu())
            t.export(sim_pts_root / f'{cur_frame * cfg.dataset.data.skip_frames:03d}.ply')
            
            for view in guidance.views:
                cam = guidance.getCameras(view, cur_frame)

                render = diff_rasterization(
                    means3D, deform_grad, gaussians,
                    cam, background,
                    scaling_modifier=cfg.gaussians.scaling_modifier,
                    force_mask_data=force_mask_data
                )
                gt = cam.original_image.to(x.device)
                

                save_image(render, sim_images_root / f"{view}_{cur_frame * cfg.dataset.data.skip_frames:03d}.png")

                
                render = render.to('cpu')
                gt = gt.to('cpu')
                
                # rgb_VBCHW[view, frame] = render
                
                if view in render_dict:
                    render_dict[view].update({
                        cur_frame: render.clone().detach()
                    })
                else:
                    render_dict.update({
                        view: {cur_frame: render.clone().detach()}
                    }) 
                
                if view in gt_dict:
                    gt_dict[view].update({
                        cur_frame: gt.clone().detach()
                    })
                else:
                    gt_dict.update({
                        view: {cur_frame: gt.clone().detach()}
                    })
                
                # renders.append(render)
                # gts.append(gt)
                

            de_x_prev = de_x.clone().detach()
            g_prev = means3D.clone().detach()
        
        
    save_render_gt_video(render_dict, gt_dict, exp_root, fps=30)
    
    # renders = torch.stack(renders).to('cuda')
    # gts = torch.stack(gts).to('cuda')
    

    # metrics = guidance.evaluate(renders, gts)
    # loss = metrics['rgb_loss']
    # print(f"metrics: {metrics['rgb_loss']*10000:.6f}, SSIM: {metrics['ssim']:.6f}, PSNR: {metrics['psnr']:.6f}")
    mean_psnr, mean_ssim, mean_lpips = 0.0, 0.0, 0.0
    for view in cfg.dataset.data.used_views:
        renders = torch.stack([frame for frame in render_dict[view].values()]).to('cuda')
        gts = torch.stack([gt for gt in gt_dict[view].values()]).to('cuda')
        metrics = evaluate(renders, gts)
        print(f"view {view} metrics: {metrics['rgb_loss'] * 10000:.6f}, {metrics['ssim']:.6f}, {metrics['psnr']:.6f}")
        mean_psnr += metrics['psnr']
        mean_ssim += metrics['ssim']
    mean_psnr /= len(cfg.dataset.data.used_views)
    mean_ssim /= len(cfg.dataset.data.used_views)
    mean_lpips /= len(cfg.dataset.data.used_views)
    print(f"mean metrics: {mean_psnr:.6f}, {mean_ssim:.6f}, {mean_lpips:.6f}")
    return mean_psnr, mean_ssim, mean_lpips
    
        
if __name__ == '__main__':        

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiment/configs/default_neuma_elastic.yaml')
    parser.add_argument('--scene_name', type=str, default='ClayCat')
    parser.add_argument('--material_model_path', type=str, default='results-model/ablation_diff_views/view2/cc/model100.pth')
    args = parser.parse_args()

    forward(args.config, args.scene_name, args.material_model_path)