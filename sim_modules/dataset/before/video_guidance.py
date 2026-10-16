from typing import Union, List
from jaxtyping import Float, Int

import os
import cv2

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
# from lpipsPyTorch import lpips


FLOW_HEIGHT = 256
FLOW_WIDTH = 480

def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy image to a PyTorch tensor.
    """
    if images.ndim == 3:
        images = images[None, ...]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

def preprocess(frames, batch_size=10, height=FLOW_HEIGHT, width=FLOW_WIDTH):

    new_batches = torch.zeros(len(frames), 3, height, width).to(frames.device)
    for start_idx in range(0, len(frames), batch_size):
        curr_batch_size = min(batch_size, len(frames) - start_idx)
        batch = frames[start_idx:start_idx+curr_batch_size]
        batch = (batch * 2. - 1.).clamp(-1.0, 1.0)
        batch = F.interpolate(
                batch, (height, width), mode="bilinear", align_corners=False
            )
        new_batches[start_idx:start_idx+curr_batch_size] = batch
    return new_batches


class VideoGuidance:
    class GuidanceInfo:
        def __init__(self, frames=None, masks=None, flows=None):
            self.frames = frames
            self.masks = masks
            self.flows = flows
        
    def __init__(self, cfg: DictConfig, downsample=1., num_frames=8, readCameras=True, use_ssim=False, use_flow=False):
        self.eval = cfg.eval
        self.cameras = {}
        self.device = cfg.device
        self._init_x = None
        self._init_v = None
        self._velocity_opt = None
        self._velocity_sch = None
        self.skip_frames = cfg.data.skip_frames
        self.use_ssim = use_ssim
        self.use_flow = use_flow
        # self.device = device
        
        self.weights_dtype = torch.float32
        self.FLOW_HEIGHT = 256
        self.FLOW_WIDTH = 480
        
        self.from_frame = cfg.data.from_frame if cfg.data.from_frame is not None else 0
        self.to_frame = cfg.data.to_frame if cfg.data.to_frame is not None else cfg.data.nframes
        

        self.save_guidance_path = Path(os.path.join(cfg.data.path, "guidance"))

        shutil.rmtree(self.save_guidance_path, ignore_errors=True)
        self.save_guidance_path.mkdir(parents=True, exist_ok=True)
        

        os.mkdir(os.path.join(str(self.save_guidance_path), "frames"))
        os.mkdir(os.path.join(str(self.save_guidance_path), "masks"))
        os.mkdir(os.path.join(str(self.save_guidance_path), "flows"))
        
        # load frames
        if readCameras:
            self.readCameras(cfg)
            

        """
        但是目前就先按照不用分割的写吧，后续再进行调整
        如果包含第0帧，那么光流预测的帧数是num_frames-1。例如400帧
        """
        

        # self.raft_model = raft_large(pretrained=True, progress=False).to(self.device)
        # self.raft_model = self.raft_model.eval()
        self.raft_model = None
        
        
        self.guidance_infos = []

        for view, cam_dict in self.cameras.items():
            # 1. get original frames

            cam_steps = sorted(cam_dict.keys(), key=lambda x: int(x))
            frames = [cam_dict[step].original_image for step in cam_steps]
            frames = torch.stack(frames, dim=0).to(self.device)
            
            if self.use_flow:
                # 2. predict optical flow
                prev_batch = preprocess(frames[:-1])
                curr_batch = preprocess(frames[1:])

                with torch.no_grad():
                    guidance_flows = self.model(prev_batch, curr_batch)[-1]
            else:
                guidance_flows = None
            
            guidance_frames = frames[1:]
            self.guidance_infos.append(self.GuidanceInfo(guidance_frames, None, guidance_flows))
            
            
            if self.use_flow:

                save_frames = []
                save_flows = []
                for idx, (frame, flow) in enumerate(zip(guidance_frames, guidance_flows)):
                    frame_img = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                    cv2.imwrite(os.path.join(str(self.save_guidance_path), "frames", f"{view}_{idx:03d}.png"), cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))

                    flow_img = flow_to_image(flow) # shape: (3, H, W), uint8, RGB

                    flow_img = flow_img.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                    cv2.imwrite(os.path.join(str(self.save_guidance_path), "flows", f"{view}_{idx:03d}.png"), flow_img)
                    
                    save_frames.append(frame_img)
                    save_flows.append(flow_img)
                    

                save_video(save_frames, os.path.join(str(self.save_guidance_path), "frames", f"{view}.mp4"))
                save_video(save_flows, os.path.join(str(self.save_guidance_path), "flows", f"{view}.mp4"))
            
                
        # print(f"views: {self.views}, steps: {self.steps}")
        
        # fnames = os.listdir(guidance_path)
        # fnames.sort()
        # frame_per_stage = len(fnames) // num_frames
        
        # frames = []
        # for i, fn in enumerate(fnames):

        #         continue

        #     if downsample != 1.:
        #         frame = cv2.resize(frame, (frame.shape[1]*downsample, frame.shape[0]*downsample)) 


        #     frames.append(frame)
            
        # self.guidance = torch.cat(frames, dim=0).to(self.device) # (BCHW)

        # # predict optical flow
        # prev_batch = preprocess(self.guidance[:-1])
        # curr_batch = preprocess(self.guidance[1:])
        # self.model = raft_large(pretrained=True, progress=False).to(self.device)
        # self.model = self.model.eval()

        # with torch.no_grad():

        
        # load masks -- PAC-NeRF
        # self.guidance_masks = None
        # mask_path = guidance_path.replace(guidance_path.split('/')[-1], 'masks')
        # if os.path.exists(mask_path):
        #     fnames = os.listdir(mask_path)
        #     fnames.sort()
            
        #     frames = []
        #     for i, fn in enumerate(fnames):
        #         if i % frame_per_stage != 0:
        #             continue
        #         frame = cv2.imread(os.path.join(mask_path, fn), cv2.IMREAD_UNCHANGED)
        #         if downsample != 1.:
        #             frame = cv2.resize(frame, (frame.shape[1]*downsample, frame.shape[0]*downsample)) 
        #         frame = numpy_to_pt(frame[..., -1:] / 255.)
        #         frames.append(frame)
        #     self.guidance_masks = torch.cat(frames, dim=0).to(self.device) # (BCHW)
        #     print("guidance_masks shape:",self.guidance_masks.shape)


    def predict_flow(self, rgb_BCHW, rgb_BCHW_init):
        # predict optical flow
        prev_batch = preprocess(torch.concat([rgb_BCHW_init, rgb_BCHW[:-1]], dim=0)).detach()
        curr_batch = preprocess(rgb_BCHW)

        rgb_flows = self.model(prev_batch, curr_batch)[-1]
        ## debug
        flow_imgs = flow_to_image(rgb_flows)

        return rgb_flows, flow_imgs

    def normalize_flow(self, flow):
        max_norm = torch.sum(flow**2, dim=1).sqrt().max()
        epsilon = torch.finfo((flow).dtype).eps
        normalized_flow = flow / (max_norm + epsilon)
        return normalized_flow

    @torch.no_grad()
    def get_raft_flow(self, frames, reshape_size=None, raft_batchsize=10):

        print("Loading RAFT model for flow estimation...")
        raft_model = raft_large(pretrained=True, progress=False).to(self.device)
        raft_model.eval()
        
        device = self.device
        weights = Raft_Large_Weights.DEFAULT
        transforms = weights.transforms()

        current_frames, next_frames = self.preprocess(frames[:-1], frames[1:], transforms, reshape_size=frames.shape[-2:])
        current_frames, next_frames = current_frames.to(device), next_frames.to(device)

        predicted_flows = []
        for i in range(0, len(current_frames), raft_batchsize):
            current_frame, next_frame = current_frames[i : i + raft_batchsize], next_frames[i : i + raft_batchsize]
            list_of_flows = raft_model(current_frame, next_frame)
            predicted_flows.append(list_of_flows[-1])
            del list_of_flows
        predicted_flows = torch.cat(predicted_flows)
        

        del raft_model
        torch.cuda.empty_cache()
        print("RAFT model unloaded.")
        
        return predicted_flows

    def preprocess(self, img1_batch, img2_batch, transforms, reshape_size):
        img1_batch = tvF.resize(img1_batch, size=reshape_size, antialias=False)
        img2_batch = tvF.resize(img2_batch, size=reshape_size, antialias=False)
        return transforms(img1_batch, img2_batch)

    # def __call__(
    #     self,
    #     rgb_VBCHW: Float[Tensor, "V B C H W"],
    #     rgb_VBCHW_init: Float[Tensor, "V B C H W"],

    #     end_frame: int,


    # ):
    #     """


    #     """
        
    #     loss = {}        
    #     loss['loss_rgb'] = 0.0
    #     loss['loss_flow'] = 0.0
        
    #     for view in range(self.valid_num_views):
    #         rgb_BCHW = rgb_VBCHW[view, ...]  # B * C * H * W
    #         rgb_BCHW_init = rgb_VBCHW_init[view, ...] # 1 * C * H * W
    #         guidance_BCHW = self.guidance_infos[view].frames[start_frame:end_frame] # B * C * H * W

    #         loss_rgb = F.mse_loss(rgb_BCHW, guidance_BCHW, reduction=reduction)
    #         loss['loss_rgb'] += loss_rgb            
            
    #         if self.use_flow:

    #             guidance_flows = self.guidance_infos[view].flows[start_frame:end_frame] # B * 2 * H * W
    #             loss_flow = F.mse_loss(rgb_flows, guidance_flows, reduction=reduction)
    #             loss['loss_flow'] += loss_flow
        

    #     loss['loss_rgb'] /= self.valid_num_views  
    #     loss['loss_flow'] /= self.valid_num_views
        
    #     return loss
    
    
    def __call__(
        self,
        renders_BCHW: Float[Tensor, "B C H W"],
        gts_BCHW: Float[Tensor, "B C H W"]
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
        
        if self.use_ssim:
            ssim_val = ssim(renders_BCHW, gts_BCHW)
            loss['loss_ssim'] += (1 - ssim_val)
        
        return loss
    
    
    def evaluate(self, renders_BCHW, gts_BCHW, use_lpips=False):
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
    
    
    def readCameras(self, cfg: Config):
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
    
    def get_key_frames(self, view=0):
        return torch.cat([self.getCameras(self.views[view], frame//self.skip_frames).original_image.unsqueeze(0) for frame in range(self.from_frame, self.to_frame, self.skip_frames)]) # shape (Views, 1, C, H, W)

    # def Cameras_to_device(self, device="cuda"):
    #     """

    #     """
    #     for view in self.cameras:
    #         for step in self.cameras[view]:
    #             self.cameras[view][step].to_device(device)

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
    def getVelocityOptimizer(self) -> torch.optim.Optimizer:
        return self._velocity_opt
    
    @property
    def getVelocityScheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self._velocity_sch
    
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
    
    
    
    def init_velocity_optimizer(self, optimizer: torch.optim.Optimizer, lr: float):
        self._velocity_opt = optimizer([self._init_v], lr=lr)
    
    def init_velocity_scheduler(self, scheduler_config: DictConfig, init_lr: float):
        self._velocity_sch = fetch_scheduler(scheduler_config).get_scheduler(self._velocity_opt, init_lr)
    
    def free_velocity_optimizer(self):
        self._velocity_opt = None
    
    def free_velocity_scheduler(self):
        self._velocity_sch = None
    
    def freeze_velocity(self):
        self._init_v.requires_grad = False
        
        
        
if __name__ == "__main__":
    import argparse
    import sim_modules.config
    from sim_modules.utils import parse_yaml_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=f'demo/invent_neohookean_from_linear/0000')
    parser.add_argument('--config', type=str, default="experiment/configs/finetune-bb.yaml")
    args, unknown_args = parser.parse_known_args()
    unknown_args = unknown_args + parse_yaml_config(args.config)
    cfg = sim_modules.config.VisionConfig(path=args.path)
    cfg.update(unknown_args)
    
    video_guidance = VideoGuidance(cfg.dataset)