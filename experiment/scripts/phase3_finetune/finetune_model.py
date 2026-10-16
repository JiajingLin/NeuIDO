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
from sim_modules.sim import MPMModelBuilder, MPMStateInitializer, MPMStaticsInitializer, MPMInitData, MPMCacheDiffSim, MPMDataset, VolumeElasticity, SigmaElasticity
from sim_modules.sim import MPMDiffSim

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
from sim_modules.dataset.video_guidance import VideoGuidance
import copy
import shutil
import omegaconf
import logging
from operator_modules.trunck_modules.trunck import ElasticityTrunckNetwork, PlasticityTrunckNetwork
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from sim_modules.dataset.vision_guidance import VisionGuidance, compute_observation_loss
from operator_modules.branch_modules.branch import VideoBranchEncoder
from torchvision.models.optical_flow import raft_large
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
        steps_per_frame = cfg.infer.phase_model_finetune.steps_per_frame
        frames_per_stage = cfg.infer.phase_model_finetune.frames_per_stage
        assert num_frames % frames_per_stage == 0, "guidance.num_frames must be divisible by cfg.optim.lower.frames_per_stage"
        stages_per_epoch = num_frames // frames_per_stage
    
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

            for frame in range(frames_per_stage):

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

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch_device = torch.device(f'cuda:{cfg.gpu}')
    torch.backends.cudnn.benchmark = True
    sim_modules.warp.replace_torch_svd()
    sim_modules.warp.replace_torch_polar()
    sim_modules.warp.replace_torch_trace()
    sim_modules.warp.replace_torch_cbrt()
    # torch.set_default_device(torch_device)

    local_dir = cfg.local_dir
    save_name = cfg.infer.phase_model_finetune.save_name
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
    
    # loss
    # pixel_loss = PIXEL_LOSSES[cfg.loss.pixel_loss]
    
    ##### Load Dataset #####
    print("Loading RAFT model...")
    raft_model = raft_large(pretrained=True, progress=False).to('cuda')
    raft_model.eval()
    guidance: VisionGuidance = VisionGuidance(cfg, raft_model=raft_model)
    del raft_model
    torch.cuda.empty_cache()
    print("RAFT model unloaded.")
    ##### Load Dataset #####
    
    #################################################################
    # Define Constutitve Model Starting
    #################################################################
    # Load the trajectory latent ##
    latent_dim = cfg.operator.latent_dim
    
    motion_encoder: VideoBranchEncoder = VideoBranchEncoder(cfg)
    motion_encoder.to(torch_device)
    motion_encoder.requires_grad_(False)
    motion_encoder.eval()
    
    if cfg.infer.pretrained_branch is not None:

        state_dict = torch.load(cfg.infer.pretrained_branch, map_location=torch_device)




        missing_keys, unexpected_keys = motion_encoder.load_state_dict(state_dict, strict=False)

        print(f"Missing keys (VAE parts): {len(missing_keys)}")

        if len(unexpected_keys) > 0:
            print(f"Warning: Unexpected keys found: {unexpected_keys}")
            


    flows = guidance.flows[guidance.views[0]].to(torch_device).unsqueeze(0)
    with torch.no_grad():
        motion_elasticity_latent_tensor, motion_plasticity_latent_tensor = motion_encoder(flows)
        



    # motion_elasticity_latent_tensor = torch.cat(motion_elasticity_latent_list, dim=0)
    # motion_plasticity_latent_tensor = torch.cat(motion_plasticity_latent_list, dim=0)

    motion_elasticity_latent = torch.nn.Embedding(motion_elasticity_latent_tensor.shape[0], latent_dim).to(torch_device)
    motion_plasticity_latent = torch.nn.Embedding(motion_plasticity_latent_tensor.shape[0], latent_dim).to(torch_device)
    motion_elasticity_latent.weight.data = motion_elasticity_latent_tensor
    motion_plasticity_latent.weight.data = motion_plasticity_latent_tensor

    ## Load the model ##
    elasticity: nn.Module = ElasticityTrunckNetwork(cfg.operator.trunck.elasticity, latent_dim)
    elasticity.to(torch_device)
    elasticity.requires_grad_(True)
    elasticity.train(True)

    plasticity: nn.Module = PlasticityTrunckNetwork(cfg.operator.trunck.plasticity, latent_dim)
    plasticity.to(torch_device)
    plasticity.requires_grad_(True)
    plasticity.train(True)
        
    if cfg.infer.pretrained_trunk is not None:
        ckpt = torch.load(cfg.infer.pretrained_trunk, map_location=torch_device)

        elasticity.load_state_dict(ckpt["elasticity"])
        plasticity.load_state_dict(ckpt["plasticity"])

        print(f"Loaded pretrained trunk from {cfg.infer.pretrained_trunk}")
    else:
        print("No pretrained trunk found, initializing from scratch.")
            
    ### lora ###
    #    -- init lora
    elasticity.init_lora_layers(r=cfg.operator.trunck.lora.r, lora_alpha=cfg.operator.trunck.lora.alpha)
    plasticity.init_lora_layers(r=cfg.operator.trunck.lora.r, lora_alpha=cfg.operator.trunck.lora.alpha)
    
    #    -- only tune lora weights
    elasticity.freeze_all_except_lora()
    plasticity.freeze_all_except_lora()
    ### lora ###
    
    ## Optimizer and scheduler ##
    elasticity_optimizer = torch.optim.AdamW(
        [{"params": motion_elasticity_latent.parameters(), "lr": cfg.finetune.elasticity_latent_lr, "weight_decay": cfg.finetune.elasticity_wd},
         {"params": filter(lambda p: p.requires_grad, elasticity.parameters()), 
          "lr": cfg.finetune.elasticity_lr, 
          "weight_decay": cfg.finetune.elasticity_wd},
         ]
    )
    elasticity_scheduler = fetch_scheduler(cfg.finetune.elasticity_scheduler).get_scheduler(elasticity_optimizer, cfg.finetune.elasticity_lr)
    
    plasticity_optimizer = torch.optim.AdamW(
        [{"params": motion_plasticity_latent.parameters(), "lr": cfg.finetune.plasticity_latent_lr, "weight_decay": cfg.finetune.plasticity_wd},
         {"params": filter(lambda p: p.requires_grad, plasticity.parameters()), 
          "lr": cfg.finetune.plasticity_lr, 
          "weight_decay": cfg.finetune.plasticity_wd}
         ]
    )
    plasticity_scheduler = fetch_scheduler(cfg.finetune.plasticity_scheduler).get_scheduler(plasticity_optimizer, cfg.finetune.plasticity_lr)
    #################################################################
    # Define Constitutive Model Ending
    #################################################################
        
    # training

    num_epochs = cfg.infer.phase_model_finetune.total_epochs

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
            for param in elasticity.parameters():
                if param is not None and param.grad is not None:
                    torch.nan_to_num_(param.grad, 0.0, 0.0, 0.0)
            clip_grad_norm_(elasticity.parameters(), 1.0, error_if_nonfinite=True)
            for param in plasticity.parameters():
                if param is not None and param.grad is not None:
                    torch.nan_to_num_(param.grad, 0.0, 0.0, 0.0)
            clip_grad_norm_(plasticity.parameters(), 1.0, error_if_nonfinite=True)
            elasticity_optimizer.step()
            plasticity_optimizer.step()
            if cfg.finetune.warmup_step == 0 or epoch > cfg.finetune.warmup_step:
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
                logger.info(f"Epoch {epoch}, view {scene_view} RGB Loss: {metrics['rgb_loss'].item() * 10000:.6f}, SSIM: {metrics['ssim'].item():.6f}, PSNR: {metrics['psnr'].item():.6f}")       
            logger.info(f"Epoch {epoch}, lr: elasticity_latent_lr: {elasticity_scheduler.get_last_lr()[0]:.6f}, plasticity_latent_lr: {plasticity_scheduler.get_last_lr()[0]:.6f}")
            logger.info(f"Epoch {epoch}, lr: elasticity_lr: {elasticity_scheduler.get_last_lr()[1]:.6f}, plasticity_lr: {plasticity_scheduler.get_last_lr()[1]:.6f}")  
            t.set_postfix(l_eval=f'{loss_eval/len(renders_dict.keys()):.6f}')
        
        save_render_gt_video_multi_scene(renders_dict, gts_dict, os.path.join(exp_root, f'epoch_{epoch}'), fps=30)
        
        # torch.cuda.empty_cache()
            

        if epoch % 1 == 0:

            torch.save({
            'epoch': epoch,
            'motion_elasticity_latent': motion_elasticity_latent.state_dict(),
            'motion_plasticity_latent': motion_plasticity_latent.state_dict(),
            'elasticity': elasticity.lora_state_dict(),
            'plasticity': plasticity.lora_state_dict(),
            }, f"{local_dir}/{save_name}/model{epoch}.pth")
            
        if epoch == num_epochs:
            t.refresh()
            break
        
    t.close()
    writer.close()
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiment/configs/default_neuma.yaml')
    parser.add_argument('--scene_name', type=str, default='RubberPawn')

    
    args = parser.parse_args()
    train(args.config, args.scene_name)

