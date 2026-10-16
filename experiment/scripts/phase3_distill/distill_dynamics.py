from collections import OrderedDict
import numpy as np
import hydra
import omegaconf
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import logging
from einops.layers.torch import Rearrange
import sys
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle as pkl
import time
import random
import sys, ipdb, traceback, subprocess
from operator_modules.trunck_modules.trunck import ElasticityTrunckNetwork, PlasticityTrunckNetwork
import argparse
import warp as wp
from operator_modules.branch_modules.branch import VideoBranchEncoder
import json
from sim_modules.dataset.video_guidance import VideoGuidance
from sim_modules.dataset.vision_guidance import VisionGuidance
from torchvision.utils import flow_to_image
from einops import rearrange
from torchvision.models.optical_flow import raft_large
from sim_modules.utils.loss_utils import nmse_loss

torch.autograd.set_detect_anomaly(True)


class F_dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.scene_cfg_base = cfg.scene_cfg_base
        self.dataset_info_path = cfg.train.phase_distill.dataset_info_path
        self.scene_l = cfg.train.phase_distill.scene_l
        self.load_data()

    def load_data(self):
        self.all_data_list = []
        
        start_time = time.time()
        

        # print("Loading RAFT model for flow estimation...")
        raft_model = raft_large(pretrained=True, progress=False).to('cuda')
        raft_model.eval()
        # raft_model = None
        

        with open(self.dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        
        for idx, scene_name in enumerate(self.scene_l):
            scene_config_path = os.path.join(self.scene_cfg_base, f"{scene_name}.yaml")
            scene_cfg = OmegaConf.load(scene_config_path)
            guidance: VideoGuidance = VideoGuidance(scene_cfg, raft_model=raft_model)
            
            motion_latent_path = dataset_info[scene_name]['motion_latent_path']
            motion_latent_target = torch.load(motion_latent_path, map_location='cpu')
            motion_elasticity_latent_target = motion_latent_target['motion_elasticity_latent']['weight'][0]
            motion_plasticity_latent_target = motion_latent_target['motion_plasticity_latent']['weight'][0]
            

            for view in guidance.views:
                self.all_data_list.append({
                    'flow_input': guidance.flows[view],
                    'motion_elasticity_latent_target': motion_elasticity_latent_target,
                    'motion_plasticity_latent_target': motion_plasticity_latent_target,
                    'traj_ids': idx
                })

            print("scene_name: ", scene_name, "motion_elasticity_latent_target norm: ", motion_elasticity_latent_target.norm())
            print("scene_name: ", scene_name, "motion_plasticity_latent_target norm: ", motion_plasticity_latent_target.norm())

        del raft_model
        torch.cuda.empty_cache()
        print("RAFT model unloaded.")

        print ("Loaded data in: ", time.time() - start_time, " seconds.")

    def __len__(self):
        return len(self.all_data_list)


    def __getitem__(self, sample_idx):
        sample = {'flow_input': self.all_data_list[sample_idx]['flow_input'],
                  'motion_elasticity_latent_target': self.all_data_list[sample_idx]['motion_elasticity_latent_target'],
                  'motion_plasticity_latent_target': self.all_data_list[sample_idx]['motion_plasticity_latent_target'],
                  'sample_idx': sample_idx
                  }
        return sample


# @hydra.main(config_path='configs', config_name='worldmodel')
def main(args):
    cfg = OmegaConf.load(args.config)
    cfg = DictConfig(cfg)
    
    wp.init()
    wp.config.verify_cuda = True
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})
    

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ##### Environment and Logging Setup #####
    local_dir = cfg.local_dir
    save_name = cfg.train.phase_distill.save_name
    
    dataset_info_path = cfg.train.phase_distill.dataset_info_path
    

    save_name = f"{save_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    torch_device = torch.device(f"cuda:{cfg.gpu}")
    
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
    
    with open(f"{local_dir}/{save_name}/config.yaml", 'w') as cfg_file:
        OmegaConf.save(cfg, cfg_file)


    ##### Load Dataset #####
    train_data: F_dataset = F_dataset(cfg=cfg)
    batch_size = cfg.train.phase_distill.batch_size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    print ("Dataset size: ", len(train_data))
    ##### Load Dataset #####


    ##### Model Setup #####
    motion_encoder: VideoBranchEncoder = VideoBranchEncoder(cfg)
    motion_encoder.to(torch_device)

    # motion_encoder_optimizer = torch.optim.AdamW([{"params": filter(lambda p: p.requires_grad, motion_encoder.parameters()), "lr": cfg.train.phase_distill.lr}])
    motion_encoder_optimizer = torch.optim.AdamW(
    [{"params": filter(lambda p: p.requires_grad, motion_encoder.parameters()), 
      "lr": cfg.train.phase_distill.lr, 
      "weight_decay": cfg.train.phase_distill.weight_decay}])
    step_lr_step_size=100
    motion_encoder_scheduler = torch.optim.lr_scheduler.StepLR(motion_encoder_optimizer, step_size=step_lr_step_size, gamma=0.9)
    ##### Model Setup #####
    
    
    ##### Define Loss ######
    loss_mse = nn.MSELoss()
    # loss_func = nmse_loss
    loss_func = loss_mse
    ##### Define Loss ######
    
    
    ##### Train #####
    total_epochs = cfg.train.phase_distill.total_epochs
    for epoch in range(total_epochs):
        # print ("Epoch: ", epoch)
        for idx, data in enumerate(train_loader):
            motion_encoder_optimizer.zero_grad()
            # data
            flow_input, motion_elasticity_latent_target, motion_plasticity_latent_target, sample_idx = data['flow_input'].cuda(), data['motion_elasticity_latent_target'].cuda(), data['motion_plasticity_latent_target'].cuda(), data['sample_idx'].cuda()
            

            # forward
            motion_elasticity_latent_pred, motion_plasticity_latent_pred = motion_encoder(flow_input, sample_idx)
            # loss

            loss_elasticity = loss_func(motion_elasticity_latent_pred, motion_elasticity_latent_target)
            loss_plasticity = loss_func(motion_plasticity_latent_pred, motion_plasticity_latent_target)
            loss = loss_elasticity + loss_plasticity
            # reg_loss = torch.norm(filter(lambda p: p.requires_grad, motion_encoder.parameters()), dim=-1)
            # reg_loss = 0.0001 * torch.mean(reg_loss)
            # loss = mse_loss + reg_loss
            
            loss.backward()
            motion_encoder_optimizer.step()
            motion_encoder_scheduler.step()

        if epoch % 100 == 0:
            elastic_mse_loss = loss_mse(motion_elasticity_latent_pred, motion_elasticity_latent_target)
            plasticity_mse_loss = loss_mse(motion_plasticity_latent_pred, motion_plasticity_latent_target)
            logger.info(f"Epoch: {epoch}, Total: {total_epochs}, Elasticity MSE Loss: {elastic_mse_loss}, Plasticity MSE Loss: {plasticity_mse_loss}, Loss: {loss}, LR1: {motion_encoder_scheduler.get_last_lr()}")

            state_dict = motion_encoder.state_dict()
            trainable_params = [n for n, p in motion_encoder.named_parameters() if p.requires_grad]
            trainable_state_dict = {k: v for k, v in state_dict.items() if k in trainable_params}
            torch.save(trainable_state_dict, f"{local_dir}/{save_name}/motion_encoder_{epoch}.pth")

        # train_loader.dataset.load_data()
        
    ##### Train #####

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="experiment/configs/default_neuma.yaml",
        help="Path to the config file."
    )
    args = parser.parse_args()
    main(args)








