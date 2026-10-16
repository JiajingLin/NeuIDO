from typing import Union, List
from jaxtyping import Float, Int

import os
import cv2
import math

import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image
from torch import Tensor
from torch import nn
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Union
# from modules.tune.scheduler import fetch_scheduler
from d3gs.scene.dataset_readers import (
    readRealCaptureCameras,
    readNeuMASyntheticCameras,
    readPACNeRFCameras,
)
from d3gs.scene.cameras import PhysCamera
from d3gs.utils.camera_utils import physCameraList_from_camInfos

import dataclasses
from sim_modules.utils import Config
import time
from pathlib import Path
import shutil
from sim_modules.utils.visualization_helper import save_video
from d3gs.utils.loss_utils import l2_loss, ssim, psnr
import torchvision.transforms.functional as tvF
from einops import rearrange
# from lpipsPyTorch import lpips


FLOW_HEIGHT = 256
FLOW_WIDTH = 480


def compute_observation_loss(
    renders_BCHW: Float[Tensor, "B C H W"],
    gts_BCHW: Float[Tensor, "B C H W"],
    use_ssim: bool = False
):
    """
    renders: 视角数 * 帧数 渲染的图像
    gts: 视角数 * 帧数 真实图像
    """
    
    loss = {}        
    loss['loss_rgb'] = 0.0
    loss['loss_ssim'] = 0.0
    # loss['loss_flow'] = 0.0
    
    # for view in range(self.valid_num_views):

    loss_rgb = l2_loss(renders_BCHW, gts_BCHW)
    loss['loss_rgb'] += loss_rgb
    
    if use_ssim:
        ssim_val = ssim(renders_BCHW, gts_BCHW)
        loss['loss_ssim'] += (1 - ssim_val)
    
    return loss

def evaluate(renders_BCHW, gts_BCHW, use_lpips=False):
    metrics = {}
    
    rgb_loss = []
    psnrs = []
    ssims = []
    lpipss = []

    for i in range(len(renders_BCHW)):

        render = renders_BCHW[i].unsqueeze(0)
        gt = gts_BCHW[i].unsqueeze(0)
        
        rgb_loss.append(l2_loss(render, gt))
        psnrs.append(psnr(render, gt))
        ssims.append(ssim(render, gt))
        
        if use_lpips:
            lpipss.append(lpips(render, gt, net_type='vgg'))
        
    metrics['rgb_loss'] = torch.tensor(rgb_loss).mean()
    metrics['psnr'] = torch.tensor(psnrs).mean()
    metrics['ssim'] = torch.tensor(ssims).mean()
    if use_lpips:
        metrics['lpips'] = torch.tensor(lpipss).mean()
    

    # metrics['rgb_loss'] = l2_loss(renders_BCHW, gts_BCHW).mean()
    """注意：rgb_loss 计算方式不同，浮点数可能精度可能有偏差，无法避免"""
    
    # psnrs = psnr(renders_BCHW, gts_BCHW).mean()
    # ssims = ssim(renders_BCHW, gts_BCHW).mean()
    # if use_lpips:
    #     lpipss = lpips(renders_BCHW, gts_BCHW, net_type='vgg')
    return metrics

def resize_for_raft(frames, max_hw=512):
    # frames: [T, C, H, W]
    T, C, H, W = frames.shape

    scale = min(1.0, max_hw / max(H, W))
    new_h = int(math.floor(H * scale / 8) * 8)
    new_w = int(math.floor(W * scale / 8) * 8)

    frames = F.interpolate(
        frames,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )
    return frames

class VideoGuidance:
    def __init__(self, cfg: DictConfig, raft_model=None):
        self.eval = cfg.dataset.eval
        self.cameras = {}
        self.device = cfg.dataset.device
        self._init_x = None
        self._init_v = None
        self.skip_frames = cfg.dataset.data.skip_frames
        
        self.weights_dtype = torch.float32
        self.FLOW_HEIGHT = 256
        self.FLOW_WIDTH = 480
        
        self.from_frame = cfg.dataset.data.from_frame if cfg.dataset.data.from_frame is not None else 0
        self.to_frame = cfg.dataset.data.to_frame if cfg.dataset.data.to_frame is not None else cfg.dataset.data.nframes
    

        self.save_guidance_path = Path(os.path.join(cfg.dataset.data.path, "guidance"))

        shutil.rmtree(self.save_guidance_path, ignore_errors=True)
        self.save_guidance_path.mkdir(parents=True, exist_ok=True)
        

        os.mkdir(os.path.join(str(self.save_guidance_path), "frames"))
        os.mkdir(os.path.join(str(self.save_guidance_path), "flows"))
    
        # load frames
        self.readCameras(cfg.dataset)
    

        self.flows = {}
        if raft_model is not None:
            for view in self.views:
                frame_TCHW = self.get_key_frames(view)
                flow_TCHW = self.get_raft_flow(frame_TCHW, raft_model=raft_model)
                flow_TCHW = flow_to_image(flow_TCHW)
                # flow_TCHW = flow_TCHW.clone().to(self.device) # shape (T, C, H, W)
                flow_CTHW = (
                    rearrange(flow_TCHW / 255.0 * 2 - 1, "T C H W -> C T H W").contiguous().to(torch.bfloat16)
                ).cpu()
                self.flows[view] = flow_CTHW
                


                save_frames = []
                save_flows = []
                for idx, (frame, flow) in enumerate(zip(frame_TCHW.clone().cpu(), flow_TCHW.clone().cpu())):
                    frame_img = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                    # cv2.imwrite(os.path.join(str(self.save_guidance_path), "frames", f"{view}_{idx:03d}.png"), cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))

                    # flow_img = flow_to_image(flow) # shape: (3, H, W), uint8, RGB

                    flow_img = flow.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                    # cv2.imwrite(os.path.join(str(self.save_guidance_path), "flows", f"{view}_{idx:03d}.png"), cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB))
                    
                    save_frames.append(frame_img)
                    save_flows.append(flow_img)

                save_video(save_frames, os.path.join(str(self.save_guidance_path), "frames", f"{view}.mp4"))
                save_video(save_flows, os.path.join(str(self.save_guidance_path), "flows", f"{view}.mp4"))

    @torch.no_grad()
    def get_raft_flow(self, frames, reshape_size=None, raft_batchsize=10, raft_model=None):

        # print("Loading RAFT model for flow estimation...")
        if raft_model is None:
            raft_model = raft_large(pretrained=True, progress=False).to(self.device)
        raft_model.eval()
        
        device = self.device
        weights = Raft_Large_Weights.DEFAULT
        transforms = weights.transforms()
        
        _, _, H, W = frames.shape
        if (H == 1080 and W == 1920) or (H == 1920 and W == 1080):
            frames = resize_for_raft(frames, max_hw=512)

        current_frames, next_frames = self.preprocess(frames[:-1], frames[1:], transforms, reshape_size=frames.shape[-2:])
        print(f"current_frames norm: {current_frames.norm(dim=1).mean()}")
        print(f"next_frames norm: {next_frames.norm(dim=1).mean()}")
        current_frames, next_frames = current_frames.to(device), next_frames.to(device)

        predicted_flows = []
        for i in range(0, len(current_frames), raft_batchsize):
            current_frame, next_frame = current_frames[i : i + raft_batchsize], next_frames[i : i + raft_batchsize]
            list_of_flows = raft_model(current_frame, next_frame)
            predicted_flows.append(list_of_flows[-1])
            del list_of_flows
        predicted_flows = torch.cat(predicted_flows)
        

        # del raft_model
        # torch.cuda.empty_cache()
        # print("RAFT model unloaded.")
        
        return predicted_flows

    def preprocess(self, img1_batch, img2_batch, transforms, reshape_size):
        img1_batch = tvF.resize(img1_batch, size=reshape_size, antialias=False)
        img2_batch = tvF.resize(img2_batch, size=reshape_size, antialias=False)
        return transforms(img1_batch, img2_batch)
    
    
    def readCameras(self, cfg: DictConfig):
        self.cameras = {}
        mode = "Training" if not self.eval else "Testing"
        camera_type = cfg.camera_type
        read_fn = eval(f"read{camera_type}Cameras")
        print(f"Reading {mode} Data")
        
        start_time = time.time()

        if cfg.data.nframes == 200 or cfg.data.nframes == 50:
            all_steps = set(range(0, 400 + 1))
        else:
            all_steps = set(range(0, cfg.data.nframes + 1))

        valid_steps = set(range(0, cfg.data.nframes + 1, self.skip_frames))
        cfg.data.exclude_steps += list(all_steps - valid_steps)
        info = read_fn(**(cfg.data))

        
        end_time = time.time()
        print(f"Reading {mode} Data Time: {end_time - start_time} seconds")
        
        self.views = info["views"]
        self.steps = info["steps"]
        self.valid_num_frames = len(self.steps) - 1
        self.valid_num_views = len(self.views)
        self.length = len(self.views) * len(self.steps)

        print(f"Loading {mode} Cameras")
        if cfg.camera.data_device is None:
            cfg.camera.data_device = cfg.device
        print(f'Setting default device for camera data to [{cfg.camera.data_device}]')
        

        start_time = time.time()
        temp_cam_list = physCameraList_from_camInfos(info["cam_infos"], 1.0, cfg.camera)
        end_time = time.time()
        print(f"Loading {mode} Cameras Time: {end_time - start_time} seconds")
        
        for cam in temp_cam_list:  
            if cam.view in self.cameras:
                self.cameras[cam.view].update({
                    cam.step: cam
                })
            else:
                self.cameras.update({
                    cam.view: {cam.step: cam}
                })

            # {
            #     'd_0': {
            #         0: PhysCamera(view='d_0', step=0),
            #         1: PhysCamera(view='d_0', step=1),
            #         ...
            #     },
            #     'd_1': {
            #         0: PhysCamera(view='d_1', step=0),
            #         1: PhysCamera(view='d_1', step=1),
            #         ...
            #     },
            #     ...
        print(f"Loaded the Camera Set with {len(self.cameras)} views and {len(self.steps)} steps")
        if len(self.views) < 20:
            print(f"    Views: {self.views}")
        if len(self.steps) < 20:
            print(f"    Steps: {self.steps}")
            
    
    def getCameras(self, view, step) -> PhysCamera:
        if isinstance(view, int):
            view = self.views[view]
        elif isinstance(view, str):
            pass
        else:
            raise ValueError(f"view must be an integer or a string, but got {view} ({type(view)})")
        return self.cameras[view][step * self.skip_frames]

    def get_init_frame(self):
        return torch.stack([self.getCameras(view, 0).original_image.unsqueeze(0) for view in self.views]) # shape (Views, 1, C, H, W)
    
    def get_key_frames(self, view):
        return torch.cat([self.getCameras(view, frame//self.skip_frames).original_image.unsqueeze(0) for frame in range(self.from_frame, self.to_frame, self.skip_frames)]) # shape (Views, 1, C, H, W)
    
    def get_flows(self, view):
        return self.flows[view]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # modulate idx to get view and step
        idx = idx % self.length
        view_id = idx // len(self.steps)
        view = self.views[view_id]
        step = idx % len(self.steps)
        
        return self.cameras[view][step]
    
    
    def get_init_material_data(self):
        init_C = torch.zeros(self._init_x.shape[0], 3, 3).to(self._init_x.device)
        init_F = torch.eye(3).unsqueeze(0).expand(self._init_x.shape[0], 3, 3).to(self._init_x.device)
        init_S = torch.zeros(self._init_x.shape[0], 3, 3).to(self._init_x.device)
        return self.get_init_x, self.get_init_v, init_C, init_F, init_S
    
    @property
    def get_init_x(self):
        return self._init_x
    
    @property
    def get_init_v(self):
        # global init_v
        if self._init_v.ndim == 1:
            return self._init_v.unsqueeze(0).expand(self._init_x.shape[0], -1)
        # custom init_v
        elif self._init_v.ndim == 2:
            return self._init_v
    
    def export_init_x_and_v(self, path):
        data = {'init_x': self.get_init_x.cpu(), 'init_v': self.get_init_v.cpu()}
        torch.save(data, path)
        print(f'Saved initial particle data (`x` and `v`) to {path}')
    
    def set_init_x_and_v(self, init_x: Union[NDArray | torch.Tensor], init_v: Optional[Union[NDArray | torch.Tensor]]=None):
        if isinstance(init_x, np.ndarray):
            self._init_x = torch.from_numpy(init_x).to(self.device).float()  # shape (N, 3)
        elif isinstance(init_x, torch.Tensor):
            self._init_x = init_x.to(self.device).float()  # shape (N, 3)
        else:
            raise ValueError(f"init_x must be a numpy array or a torch tensor, but got {type(init_x)}")

        if init_v is None:
            init_v = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # shape (3,)
            init_v = torch.from_numpy(init_v)  # shape (3,)
            self._init_v = nn.Parameter(init_v.to(self.device), requires_grad=True)  # shape (3,)
        else:
            if isinstance(init_v, np.ndarray):
                self._init_v = torch.from_numpy(init_v).to(self.device).float()  # shape (N, 3)
            elif isinstance(init_v, torch.Tensor):
                self._init_v = init_v.requires_grad_(False).to(self.device).float()  # shape (N, 3)
            else:
                raise ValueError(f"init_v must be a numpy array or a torch tensor, but got {type(init_v)}")
        