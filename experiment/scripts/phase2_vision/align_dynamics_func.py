import sys
import dataclasses
import random
from pathlib import Path
import argparse
import math

import yaml
from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import warp as wp

import sim_modules
from sim_modules.sim import MPMModelBuilder, MPMStateInitializer, MPMStaticsInitializer, MPMInitData, MPMDiffSim, MPMDataset, VolumeElasticity, SigmaElasticity

from sim_modules.utils.config_helper import parse_yaml_config, parse_unknown_args
from d3gs.scene.gaussian_model import GaussianModel

import os

from sim_modules.utils.binding_helper import *
from d3gs.utils.render_utils import diff_rasterization
# from sim_modules.dataset import VideoDataset

from sim_modules.utils.visualization_helper import save_render_gt_video_multi_scene, save_video, save_render_video, save_render_gt_video
from d3gs.utils.loss_utils import l1_loss, l2_loss
from sim_modules.utils.eval_helper import compute_eval_metrics

from sim_modules.utils import get_root, get_script_parser, dict_to_cmds

from sim_modules.utils.local_utils import LinearStepAnneal
# from eval import eval_one_epoch
from sim_modules.dataset.vision_guidance import VisionGuidance

import copy
import shutil
import omegaconf
import logging
from operator_modules.trunck_modules.trunck import ElasticityTrunckNetwork, PlasticityTrunckNetwork
from omegaconf import DictConfig, OmegaConf
from sim_modules.tune.scheduler import fetch_scheduler


PIXEL_LOSSES = {
    "l1": l1_loss,
    "l2": l2_loss
}

@torch.enable_grad()
def train_one_traj(cfg, guidance: VisionGuidance, plasticity, elasticity, motion_elasticity_latent, motion_plasticity_latent, epoch, traj_idx=0):
        torch_device = 'cuda'
        # GS + Particles
        gaussians = guidance.gaussians
        bindings = guidance.bindings
        background = guidance.background
        force_mask_data = guidance.force_mask_data
        gs_scaling_modifier = guidance.gs_scaling_modifier
        
        # Simulator
        init_data = guidance.init_data
        diff_sim: MPMDiffSim = guidance.diff_sim

        # parameters
        num_frames = guidance.valid_num_frames
        steps_per_frame = cfg.train.phase_align.steps_per_frame
        frames_per_stage = cfg.train.phase_align.frames_per_stage
        # assert num_frames % frames_per_stage == 0, "guidance.num_frames must be divisible by cfg.optim.lower.frames_per_stage"
        # stages_per_epoch = num_frames // frames_per_stage
        stages_per_epoch = math.ceil(num_frames / frames_per_stage)
    
        renders_dict = {}
        gt_dict = {}
        


        x, v, C, F, _ = guidance.get_init_material_data()
        
        de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)

        de_x_prev = de_x.clone().detach()
        g_prev = gaussians.get_xyz.clone().detach()
        

        if cfg.finetune.lambda_max_decay > 0:
            coeff_max_decay = 1. / cfg.finetune.lambda_max_decay
            ratio = min(coeff_max_decay * epoch / cfg.finetune.num_epochs, 1.0)
        else:
            ratio = 1.0
        decay_rate = cfg.finetune.decay_init + (cfg.finetune.decay_final - cfg.finetune.decay_init) * ratio
        

        for stage in range(stages_per_epoch):
            loss_rgb = 0.0
            renders = []
            gts = []


            cur_frames_per_stage = min(frames_per_stage, num_frames - stage * frames_per_stage)
            for frame in range(cur_frames_per_stage):

                for step in range(steps_per_frame):
                    cur_step = stage * frames_per_stage * steps_per_frame + frame * steps_per_frame + step
                    
                    traj_ids = torch.full((F.shape[0],), traj_idx, dtype=torch.long, device=torch_device)
                    
                    stress = elasticity(F, motion_elasticity_latent, traj_ids=traj_ids)
                    x, v, C, F = diff_sim(x, v, C, F, stress)
                    F = plasticity(F, motion_plasticity_latent, traj_ids=traj_ids)
                    

                cur_frame = stage * frames_per_stage + frame + 1
                de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)
                """在此处实现的particle-GS binding的计算"""
                means3D = compute_bindings_xyz(de_x, de_x_prev, g_prev, bindings)
                deform_grad = compute_bindings_F(F, bindings)
                """在此处实现的particle-GS binding的计算"""                    
            
                for view in guidance.views:
                # for view in ['d_2']:
                    cam = guidance.getCameras(view, cur_frame)
                    
                    render = diff_rasterization(
                    means3D, deform_grad, gaussians,
                    cam, background,
                    scaling_modifier=gs_scaling_modifier,
                    force_mask_data=force_mask_data
                    )
                    gt = cam.original_image.to(x.device)
                    
                    # accumulate loss
                    rollout_decay_rate = decay_rate ** ((cur_step) // cfg.finetune.decay_steps)
                    loss_rgb += rollout_decay_rate * l2_loss(render, gt)
                    
                    if view in renders_dict:
                        renders_dict[view].update({
                            cur_frame: render.clone().detach()
                        })
                    else:
                        renders_dict.update({
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
                    
                de_x_prev = de_x.clone().detach() 
                g_prev = means3D.clone().detach()
            

            loss = loss_rgb
            
            loss.backward()
            




            x, v, C, F = (
                x.detach(),
                v.detach(),
                C.detach(),
                F.detach()
            )
        
        return renders_dict, gt_dict
    

@torch.enable_grad()
def train(config: str, scene_name: str):


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
    sim_modules.warp.replace_torch_svd()
    sim_modules.warp.replace_torch_polar()
    sim_modules.warp.replace_torch_trace()
    sim_modules.warp.replace_torch_cbrt()
    torch.set_default_device(torch_device)

    local_dir = cfg.local_dir
    save_name = cfg.train.phase_align.save_name
    save_name = f"{save_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_root = Path(f"{local_dir}/{save_name}")
    
    state_root = exp_root / 'state'
    ckpt_root = exp_root / 'ckpt'
    sim_modules.utils.mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
    state_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    with open(exp_root/ 'config.yaml', 'w') as cfg_file:
        OmegaConf.save(cfg, cfg_file)

    writer = SummaryWriter(exp_root, purge_step=0)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(f"{local_dir}/{save_name}/logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"{local_dir}/{save_name}/logs")
    fh = logging.FileHandler(f"{local_dir}/{save_name}/log.txt")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    fh.setLevel(logging.INFO)
    # fh.setLevel(logging.DEBUG) # or any level you want
    logger.addHandler(fh)
    
    ##### Load Dataset #####
    guidance: VisionGuidance = VisionGuidance(cfg, None)
    ##### Load Dataset #####
    
    #################################################################
    # Define Constutitve Model Starting
    #################################################################
    # Load the trajectory latent ##
    latent_dim = cfg.operator.latent_dim
    pre_elasticity_latent_path = cfg.train.phase_align.pretrained_elasticity_latent
    pre_plasticity_latent_path = cfg.train.phase_align.pretrained_plasticity_latent
    
    if pre_elasticity_latent_path is not None:
        elasticity_latent_ckpt = torch.load(pre_elasticity_latent_path, map_location=torch_device)
        print(f"Loaded elasticity latent ckpt: {elasticity_latent_ckpt}")
        nums_latent, dim = elasticity_latent_ckpt["weight"].shape
        if dim != latent_dim:
            raise ValueError(
                f"Loaded latent dim {dim} != cfg.operator.latent_dim {latent_dim}"
            )


        latent_mean = elasticity_latent_ckpt["weight"][10]
        motion_elasticity_latent = torch.nn.Embedding(1, latent_dim).to(torch_device) # shape (1, latent_dim)
        motion_elasticity_latent.weight.data = latent_mean.unsqueeze(0) # shape (1, latent_dim)
        print(f"Loaded motion_elasticity_latent: {motion_elasticity_latent.weight}")
        print(f"Loaded pretrained latent from {pre_elasticity_latent_path}")
    else:
        motion_elasticity_latent = torch.nn.Embedding(1, latent_dim).cuda()
        nn.init.normal_(motion_elasticity_latent.weight, mean=0.0, std=1.0)
        print("No pretrained elasticity latent found, initializing from scratch.")
        
        
    if pre_plasticity_latent_path is not None:
        plasticity_latent_ckpt = torch.load(pre_plasticity_latent_path, map_location=torch_device)
        print(f"Loaded plasticity latent ckpt: {plasticity_latent_ckpt}")
        nums_latent, dim = plasticity_latent_ckpt["weight"].shape
        if dim != latent_dim:
            raise ValueError(
                f"Loaded latent dim {dim} != cfg.operator.latent_dim {latent_dim}"
            )


        latent_mean = plasticity_latent_ckpt["weight"][10]
        motion_plasticity_latent = torch.nn.Embedding(1, latent_dim).to(torch_device) # shape (1, latent_dim)
        motion_plasticity_latent.weight.data = latent_mean.unsqueeze(0) # shape (1, latent_dim)
        print(f"Loaded motion_plasticity_latent: {motion_plasticity_latent.weight}")
        print(f"Loaded pretrained latent from {pre_plasticity_latent_path}")
    else:
        motion_plasticity_latent = torch.nn.Embedding(1, latent_dim).cuda()
        nn.init.normal_(motion_plasticity_latent.weight, mean=0.0, std=1.0)
        print("No pretrained plasticity latent found, initializing from scratch.")

    ## Load the model ##
    elasticity: nn.Module = ElasticityTrunckNetwork(cfg.operator.trunck.elasticity, latent_dim)
    elasticity.to(torch_device)
    elasticity.requires_grad_(False)
    elasticity.eval()

    plasticity: nn.Module = PlasticityTrunckNetwork(cfg.operator.trunck.plasticity, latent_dim)
    plasticity.to(torch_device)
    plasticity.requires_grad_(False)
    plasticity.eval()
        
    if cfg.train.phase_align.pretrained_trunk is not None:
        ckpt = torch.load(cfg.train.phase_align.pretrained_trunk, map_location=torch_device)

        elasticity.load_state_dict(ckpt["elasticity"])
        plasticity.load_state_dict(ckpt["plasticity"])

        print(f"Loaded pretrained trunk from {cfg.train.phase_align.pretrained_trunk}")
    else:
        print("No pretrained trunk found, initializing from scratch.")
    
    ## Optimizer and scheduler ##
    # operator_optimizer = torch.optim.AdamW(
    #     [{"params": motion_elasticity_latent.parameters(), "lr": cfg.train.phase_align.lr, "weight_decay": 0.0},
    #      {"params": motion_plasticity_latent.parameters(), "lr": cfg.train.phase_align.lr, "weight_decay": 0.0}])
    # operator_scheduler = None
    # operator_scheduler = fetch_scheduler('cos').get_scheduler(operator_optimizer, cfg.train.phase_align.lr)
    
    elasticity_optimizer = torch.optim.AdamW(
        [{"params": motion_elasticity_latent.parameters(), "lr": cfg.align.elasticity_latent_lr, "weight_decay": cfg.align.elasticity_wd},]
    )
    elasticity_scheduler = fetch_scheduler(cfg.align.elasticity_scheduler).get_scheduler(elasticity_optimizer, cfg.align.elasticity_latent_lr)
    
    plasticity_optimizer = torch.optim.AdamW(
        [{"params": motion_plasticity_latent.parameters(), "lr": cfg.align.plasticity_latent_lr, "weight_decay": cfg.align.plasticity_wd}]
        )
        
    plasticity_scheduler = fetch_scheduler(cfg.align.plasticity_scheduler).get_scheduler(plasticity_optimizer, cfg.align.plasticity_latent_lr)
    ## Optimizer and scheduler ##
    
    #################################################################
    # Define Constitutive Model Ending
    #################################################################
    
    
    
        
    # training

    num_epochs = cfg.train.phase_align.total_epochs 
    t = trange(num_epochs + 1, desc=f'[train] {save_name}', file=sys.stdout, leave=None)
    for epoch in t:
        elasticity_optimizer.zero_grad()
        plasticity_optimizer.zero_grad()
        
        renders_dict = {}
        gts_dict = {}
            
        renders_dict, gts_dict = train_one_traj(cfg, guidance, plasticity, elasticity, motion_elasticity_latent, motion_plasticity_latent, epoch)
        
        
        try:
            for param in motion_elasticity_latent.parameters():
                if param is not None and param.grad is not None:
                    torch.nan_to_num_(param.grad, 0.0, 0.0, 0.0)
            clip_grad_norm_(motion_elasticity_latent.parameters(), 1.0, error_if_nonfinite=True)
            for param in motion_plasticity_latent.parameters():
                if param is not None and param.grad is not None:
                    torch.nan_to_num_(param.grad, 0.0, 0.0, 0.0)
            clip_grad_norm_(motion_plasticity_latent.parameters(), 1.0, error_if_nonfinite=True)
            elasticity_optimizer.step()
            plasticity_optimizer.step()
            if elasticity_scheduler is not None and plasticity_scheduler is not None:
                elasticity_scheduler.step()
                plasticity_scheduler.step()
        except RuntimeError as e:
            tqdm.write(str(e))
            break
            
        with torch.no_grad():
            loss_eval = 0.0
            for scene_view in renders_dict.keys():
                metrics = compute_eval_metrics(renders_dict[scene_view], gts_dict[scene_view])
                loss_eval += metrics['rgb_loss'].item() * 10000
                logger.info(f"Epoch {epoch}, {scene_view} RGB Loss: {metrics['rgb_loss'].item() * 10000:.6f}, SSIM: {metrics['ssim'].item():.6f}, PSNR: {metrics['psnr'].item():.6f}")
            t.set_postfix(l_eval=f'{loss_eval/len(renders_dict.keys()):.6f}')
        
        save_render_gt_video_multi_scene(renders_dict, gts_dict, os.path.join(exp_root, f'epoch_{epoch}'), fps=cfg.video_fps)
        
        # torch.cuda.empty_cache()
                

        if epoch % 1 == 0:

            torch.save({
                'motion_elasticity_latent': motion_elasticity_latent.state_dict(),
                'motion_plasticity_latent': motion_plasticity_latent.state_dict(),
            }, ckpt_root / f'motion_latent_{epoch}.pth')
            # torch.save(motion_elasticity_latent.state_dict(), ckpt_root / f'motion_elasticity_latent_{epoch}.pth')
            # torch.save(motion_plasticity_latent.state_dict(), ckpt_root / f'motion_plasticity_latent_{epoch}.pth')
            
        if epoch == num_epochs:
            t.refresh()
            break
        
    t.close()
    writer.close()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='experiment/configs/default_neuma.yaml')
    # parser.add_argument('--scene_name', type=str, default='HoneyBottle')
    parser.add_argument('--config', type=str, default='experiment/configs/default_realworld.yaml')
    parser.add_argument('--scene_name', type=str, default='Burger')
    # parser.add_argument('--config', type=str, default='experiment/configs/default_pacnerf.yaml')
    # parser.add_argument('--scene_name', type=str, default='cat')

    
    args = parser.parse_args()
    train(args.config, args.scene_name)

