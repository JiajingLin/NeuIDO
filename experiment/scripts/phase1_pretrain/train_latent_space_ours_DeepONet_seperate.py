from collections import OrderedDict
import operator
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
from torch.utils.data import DataLoader, random_split
import pickle as pkl
import time
import random
import sys, ipdb, traceback, subprocess
from operator_modules.trunck_modules.trunck import ElasticityTrunckNetwork, PlasticityTrunckNetwork
import argparse
import warp as wp
from sim_modules.utils.loss_utils import nmse_loss
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

torch.autograd.set_detect_anomaly(True)


class F_dataset(torch.utils.data.Dataset):
    def __init__(self, file_dir_list, dataset_dir):
        self.file_dir_list = file_dir_list
        self.dataset_dir = dataset_dir
        self.load_data()

        stats = self.compute_gt_stats(self)
        self.set_normalizers(stats)

    def load_data(self):
        self.all_data_dict = {}
        self.traj_id_dict = {}

        start_time = time.time()
        for idx, file_dir in enumerate(self.file_dir_list):
            print ("Load data: ", idx)

            idx_l = list(np.arange(0, 500))
            random.shuffle(idx_l)

            cur_Ftmp = torch.load(os.path.join(f"{self.dataset_dir}/dataset", f"{file_dir}", "GtFtmp.pt"), map_location='cpu')[1:]
            cur_F = torch.load(os.path.join(f"{self.dataset_dir}/dataset", f"{file_dir}" ,"GtF.pt"), map_location='cpu')[1:]
            cur_stress = torch.load(os.path.join(f"{self.dataset_dir}/dataset", f"{file_dir}", "GtStress.pt"), map_location='cpu')[1:]
            cur_C = torch.load(os.path.join(f"{self.dataset_dir}/dataset", f"{file_dir}", "GtC.pt"), map_location='cpu')[1:]
            if cur_C.shape[0] == 960:
                cur_Ftmp = torch.cat((cur_Ftmp, cur_Ftmp, cur_Ftmp[:80, :, :, :]), dim=0)
                cur_F = torch.cat((cur_F, cur_F, cur_F[:80, :, :, :]), dim=0)
                cur_stress = torch.cat((cur_stress, cur_stress, cur_stress[:80, :, :, :]), dim=0)
                cur_C = torch.cat((cur_C, cur_C, cur_C[:80, :, :, :]), dim=0)
            cur_traj_id = torch.full((cur_F.shape[0], cur_F.shape[1]), idx)

            self.sim_timesteps = cur_Ftmp.shape[0]
            self.sim_num_particles = cur_Ftmp.shape[1]

            input_Ftmp_tensor = cur_Ftmp.reshape(-1, 3, 3)
            input_F_tensor = cur_F.reshape(-1, 3, 3)
            gt_stress_tensor = cur_stress.reshape(-1, 3, 3)
            input_C_tensor = cur_C.reshape(-1, 3, 3)
            traj_ids = cur_traj_id.reshape(-1)
            self.traj_id_dict[idx] = f"{file_dir}"
            

            max_input_F = torch.max(input_F_tensor)
            max_gt_stress = torch.max(gt_stress_tensor)
            print ("Max input_F: ", max_input_F, "Max gt_stress: ", max_gt_stress)
            min_input_F = torch.min(input_F_tensor)
            min_gt_stress = torch.min(gt_stress_tensor)
            print ("Min input_F: ", min_input_F, "Min gt_stress: ", min_gt_stress)

            self.all_data_dict[idx] = {'input_F': input_F_tensor, 'gt_stress': gt_stress_tensor, 'traj_ids': traj_ids,
                                       'input_Ftmp': input_Ftmp_tensor, 'input_C': input_C_tensor}

        print ("Loaded data in: ", time.time() - start_time, " seconds.")

    def __len__(self):
        return len(self.file_dir_list) * self.sim_num_particles * self.sim_timesteps

    def __getitem__(self, idx):
        traj_idx = idx // (self.sim_num_particles * self.sim_timesteps)
        sample_idx = idx % (self.sim_num_particles * self.sim_timesteps)

        sample = {'input_F': self.all_data_dict[traj_idx]['input_F'][sample_idx],
                  'stress_target': self.all_data_dict[traj_idx]['gt_stress'][sample_idx],
                  'input_C': self.all_data_dict[traj_idx]['input_C'][sample_idx],
                  'traj_ids': self.all_data_dict[traj_idx]['traj_ids'][sample_idx],
                  'input_Ftmp': self.all_data_dict[traj_idx]['input_Ftmp'][sample_idx],
                  }
        return sample
    
    
    def set_normalizers(self, stats):


        self.F_mean = stats['F']['mean'].to('cuda')            # (9,)
        self.F_std  = stats['F']['std'].to('cuda')             # (9,)
        self.S_mean = stats['stress']['mean'].to('cuda')       # (9,)
        self.S_std  = stats['stress']['std'].to('cuda')        # (9,)


    def z_norm_F(self, F):           # F: (...,3,3)
        x = F.reshape(-1, 9)
        x = (x - self.F_mean) / self.F_std
        return x.reshape(*F.shape)

    def z_denorm_F(self, Fn):        # Fn: (...,3,3)
        x = Fn.reshape(-1, 9)
        x = x * self.F_std + self.F_mean
        return x.reshape(*Fn.shape)

    def z_norm_stress(self, S):      # S: (...,3,3)
        x = S.reshape(-1, 9)
        x = (x - self.S_mean) / self.S_std
        return x.reshape(*S.shape)

    def z_denorm_stress(self, Sn):   # Sn: (...,3,3)
        x = Sn.reshape(-1, 9)
        x = x * self.S_std + self.S_mean
        return x.reshape(*Sn.shape)
    
    @torch.no_grad()
    def compute_gt_stats(self, train_dataset, eps: float = 1e-12):
        """
        从 train_dataset 提取所有 GT，计算逐分量的 mean/std。
        - 对 3x3 张量：先展平到 (N, 9) 再算统计量。
        - 只统计训练集！
        """

        F_list, Stress_list = [], []

        for idx in train_dataset.all_data_dict.keys():
            data = train_dataset.all_data_dict[idx]

            F_gt = data['input_F']
            S_gt = data['gt_stress']  # (N,3,3)

            F_list.append(F_gt.reshape(-1, 9))
            Stress_list.append(S_gt.reshape(-1, 9))

        F_all = torch.cat(F_list, dim=0)           # (N_total, 9)
        S_all = torch.cat(Stress_list, dim=0)      # (N_total, 9)

        F_mean = F_all.mean(dim=0)
        F_std  = F_all.std(dim=0, unbiased=False).clamp_min(eps)

        S_mean = S_all.mean(dim=0)
        S_std  = S_all.std(dim=0, unbiased=False).clamp_min(eps)

        stats = {
            'F':     {'mean': F_mean, 'std': F_std},
            'stress':{'mean': S_mean, 'std': S_std},
        }
        return stats


def evaluate(elasticity, plasticity, trajectory_elasticity_latent, trajectory_plasticity_latent, val_loader):
    elasticity.eval()
    plasticity.eval()
    
    # Collect all validation data
    all_v_input_F = []
    all_v_stress_target = []
    all_v_traj_ids = []
    all_v_input_Ftmp = []
    all_v_input_C = [] # Assuming input_C is not used in forward pass but just in case
    
    with torch.no_grad():
        for val_batch in val_loader:
            all_v_input_F.append(val_batch['input_F'].cuda())
            all_v_stress_target.append(val_batch['stress_target'].cuda())
            all_v_traj_ids.append(val_batch['traj_ids'].cuda())
            all_v_input_Ftmp.append(val_batch['input_Ftmp'].cuda())
            
        if len(all_v_input_F) == 0:
             elasticity.train()
             plasticity.train()
             return 0.0, 0.0, 0.0

        v_input_F = torch.cat(all_v_input_F, dim=0)
        v_stress_target = torch.cat(all_v_stress_target, dim=0)
        v_traj_ids = torch.cat(all_v_traj_ids, dim=0)
        v_input_Ftmp = torch.cat(all_v_input_Ftmp, dim=0)
        
        v_pre_stress = elasticity(v_input_F, trajectory_elasticity_latent, v_traj_ids)
        v_pred_Fproj = plasticity(v_input_Ftmp, trajectory_plasticity_latent, v_traj_ids)
        
        loss_func = nn.MSELoss()
        
        v_loss_l1_stress = loss_func(v_pre_stress, v_stress_target)
        v_loss_l1_fproj = loss_func(v_pred_Fproj, v_input_F) # Note: Original code used input_F for Fproj target
        
        avg_stress_loss = v_loss_l1_stress.item()
        avg_fproj_loss = v_loss_l1_fproj.item()
        avg_val_loss = avg_stress_loss + avg_fproj_loss

    elasticity.train()
    plasticity.train()
    return avg_val_loss, avg_stress_loss, avg_fproj_loss

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
    save_name = cfg.train.phase_pretrain.save_name
    
    dataset_dir = cfg.train.phase_pretrain.dataset_dir
    

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
    traj_l = cfg.train.phase_pretrain.traj_l
    traj_dir_list = []
    for tl in traj_l:
        # for _dir_idx, _dir in enumerate(os.listdir(os.path.join(f"{dataset_dir}/dataset/", tl))[:1]):
        for _dir_idx, _dir in enumerate(os.listdir(os.path.join(f"{dataset_dir}/dataset/", tl))):
            traj_dir_list.append(f"{tl}/{_dir}")

    count_traj = len(traj_dir_list)
    train_data: F_dataset = F_dataset(file_dir_list=traj_dir_list, dataset_dir=dataset_dir)
    
    # Split dataset into train and validation
    val_ratio = cfg.train.phase_pretrain.val_ratio
    val_size = int(val_ratio * len(train_data))
    train_size = len(train_data) - val_size
    
    train_subset, val_subset = random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    

    train_loader = DataLoader(train_subset, batch_size=cfg.train.phase_pretrain.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=cfg.train.phase_pretrain.batch_size, shuffle=False, drop_last=False)
    print ("Dataset size: ", len(train_data), "Train size: ", len(train_subset), "Val size: ", len(val_subset), "Number of trajectories: ", count_traj)

    ##### Model Setup #####

    # Define Latent
    latent_dim = cfg.operator.latent_dim
    trajectory_elasticity_latent = torch.nn.Embedding(count_traj, latent_dim).cuda()
    trajectory_plasticity_latent = torch.nn.Embedding(count_traj, latent_dim).cuda()
    traj_mean, traj_std = 0., 1.
    torch.nn.init.normal_(trajectory_elasticity_latent.weight, mean=traj_mean, std=traj_std)
    torch.nn.init.normal_(trajectory_plasticity_latent.weight, mean=traj_mean, std=traj_std)

    # Define Model
    elasticity: nn.Module = ElasticityTrunckNetwork(cfg.operator.trunck.elasticity, latent_dim)
    elasticity.to(torch_device)
    elasticity.requires_grad_(True)
    elasticity.train(True)

    plasticity: nn.Module = PlasticityTrunckNetwork(cfg.operator.trunck.plasticity, latent_dim)
    plasticity.to(torch_device)
    plasticity.requires_grad_(True)
    plasticity.train(True)
    
    #########################################################
    base_lr = cfg.train.phase_pretrain.lr
    optimizer = torch.optim.AdamW([{"params": elasticity.parameters(), "lr": base_lr},
                                {"params": plasticity.parameters(), "lr": base_lr},
                                {"params": trajectory_elasticity_latent.parameters(), "lr": base_lr},
                                {"params": trajectory_plasticity_latent.parameters(), "lr": base_lr}]
                                , weight_decay=cfg.train.phase_pretrain.weight_decay)
    total_epochs = cfg.train.phase_pretrain.total_epochs

    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch

    warmup_ratio = cfg.train.phase_pretrain.warmup_ratio
    warmup_steps = int(total_steps * warmup_ratio)

    warmup_start_factor = cfg.train.phase_pretrain.warmup_start_factor
    warmup_end_factor = cfg.train.phase_pretrain.warmup_end_factor
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=warmup_end_factor,
        total_iters=warmup_steps
    )


    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,

    )


    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    #########################################################
    # loss_func_stress = nn.MSELoss()
    # loss_func_fproj = nn.MSELoss()
    loss_func_stress = nmse_loss
    loss_func_fproj = nmse_loss
    total_epochs = cfg.train.phase_pretrain.total_epochs
    for epoch in range(total_epochs):
        print ("Epoch: ", epoch)
        for idx, data in enumerate(train_loader):
            input_F, stress_target, traj_ids, input_Ftmp, input_C = data['input_F'].cuda(), data['stress_target'].cuda(), data['traj_ids'].cuda(), data['input_Ftmp'].cuda(), data['input_C'].cuda()

            optimizer.zero_grad()

            pre_stress = elasticity(input_F, trajectory_elasticity_latent, traj_ids)
            pred_Fproj = plasticity(input_Ftmp, trajectory_plasticity_latent, traj_ids)

            loss_l2 = loss_func_stress(pre_stress, stress_target)
            loss_l2_fproj = loss_func_fproj(pred_Fproj, input_F)

            loss = loss_l2 + loss_l2_fproj

            loss.backward()
            optimizer.step()
            
            scheduler.step()

            if idx % 10 == 0:
                val_loss, val_stress_loss, val_fproj_loss = evaluate(elasticity, plasticity, trajectory_elasticity_latent, trajectory_plasticity_latent, val_loader)
                logger.info(f"Epoch: {epoch}, Iter/Total: {idx}/{len(train_loader)}, Val Stress: {val_stress_loss}, Val Fproj: {val_fproj_loss}, LR: {scheduler.get_last_lr()[0]}")

        if epoch % 1 == 0:
            torch.save(trajectory_elasticity_latent.state_dict(), f"{local_dir}/{save_name}/motion_elasticity_latent_{epoch}.pth")
            torch.save(trajectory_plasticity_latent.state_dict(), f"{local_dir}/{save_name}/motion_plasticity_latent_{epoch}.pth")
            
            torch.save({
                'epoch': epoch,
                'elasticity': elasticity.state_dict(),
                'plasticity': plasticity.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, f"{local_dir}/{save_name}/model_{epoch}.pth")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="experiment/configs/default_neuma.yaml",
        help="Path to the config file."
    )
    args = parser.parse_args()
    main(args)








