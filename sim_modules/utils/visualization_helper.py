import imageio
import ffmpeg
import os
import shutil
from pathlib import Path
import numpy as np
import mediapy
import cv2
# from sim_modules.utils.flow_viz import flow_to_image

# def save_video(renders, video_path, fps=30):
#     """






#     """

#     tmp_dir = Path('./temp_frames')
#     tmp_dir.mkdir(exist_ok=True)


#     for i, img in enumerate(renders):
#         frame_path = tmp_dir / f'frame_{i:04d}.png'
#         imageio.imwrite(frame_path, img)


#     (
#         ffmpeg
#         .input(f'{tmp_dir}/frame_%04d.png', framerate=fps)
#         .output(str(video_path), vcodec='libx264', pix_fmt='yuv420p')
#         .overwrite_output()
#         .run()
#     )


#     shutil.rmtree(tmp_dir)

    
    
# def save_render_video(renders_dict, vid_root):

#     for view, frame_dict in renders_dict.items():

#         sorted_frames = sorted(frame_dict.keys())
        

#         renders = [(frame_dict[frame].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for frame in sorted_frames]
        
#         os.makedirs(vid_root, exist_ok=True)
        

#         video_path = os.path.join(vid_root, f'{view}.mp4')
        

#         save_video(renders, video_path)

import os
import numpy as np
from pathlib import Path
import mediapy


def save_video(renders, video_path, fps=30):
    """
    使用 mediapy 将图像帧保存为 MP4 视频。

    参数:
        renders (List[np.ndarray]): 图像帧列表，每个为 (H, W, 3) 的 RGB 图像
        video_path (str or Path): 输出视频文件路径
        fps (int): 视频帧率
    """
    video_path = str(video_path)
    mediapy.write_video(video_path, renders, fps=fps, qp=18)


def save_render_video(renders_dict, vid_root, fps=30):
    """
    将渲染图像保存为多个视角的视频文件。

    参数:
        renders_dict (dict): 每个视角对应帧字典 {view: {frame_id: Tensor}}
        vid_root (str): 输出视频的根目录
        fps (int): 视频帧率
    """
    os.makedirs(vid_root, exist_ok=True)

    for view, frame_dict in renders_dict.items():

        sorted_frames = sorted(frame_dict.keys())


        renders = [
            (frame_dict[frame].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            for frame in sorted_frames
        ]


        video_path = os.path.join(vid_root, f'{view}.mp4')


        save_video(renders, video_path, fps=fps)

def save_render_gt_video(renders_dict, gt_dict, vid_root, fps=30):
    """
    将渲染图像保存为多个视角的视频文件。

    参数:
        renders_dict (dict): 每个视角对应帧字典 {view: {frame_id: Tensor}}
        vid_root (str): 输出视频的根目录
        fps (int): 视频帧率
    """
    os.makedirs(vid_root, exist_ok=True)

    for view, frame_dict in renders_dict.items():

        sorted_frames = sorted(frame_dict.keys())


        renders = [
            (frame_dict[frame].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            for frame in sorted_frames
        ]


        video_path = os.path.join(vid_root, f'{view}_render.mp4')


        save_video(renders, video_path, fps=fps)


    for view, frame_dict in gt_dict.items():

        sorted_frames = sorted(frame_dict.keys())


        gt = [
            (frame_dict[frame].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            for frame in sorted_frames
        ]


        video_path = os.path.join(vid_root, f'{view}_gt.mp4')


        save_video(gt, video_path, fps=fps)
        

def save_render_gt_video_multi_scene(scene_renders_dict, scene_gts_dict, vid_root, fps=30):
    """
    将渲染图像保存为多个视角的视频文件。

    参数:
        scene_renders_dict (dict): 每个场景对应视角对应帧字典 {scene_name: {view: {frame_id: Tensor}}}
        scene_gts_dict (dict): 每个场景对应视角对应帧字典 {scene_name: {view: {frame_id: Tensor}}}
        vid_root (str): 输出视频的根目录
        fps (int): 视频帧率
    """
    """
    将渲染图像保存为多个视角的视频文件。

    参数:
        renders_dict (dict): 每个视角对应帧字典 {view: {frame_id: Tensor}}
        vid_root (str): 输出视频的根目录
        fps (int): 视频帧率
    """
    os.makedirs(vid_root, exist_ok=True)

    for scene_name, renders_dict in scene_renders_dict.items():

        sorted_frames = sorted(renders_dict.keys())


        renders = [
            (renders_dict[frame].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            for frame in sorted_frames
        ]


        video_path = os.path.join(vid_root, f'{scene_name}_render.mp4')


        save_video(renders, video_path, fps=fps)


    for scene_name, gts_dict in scene_gts_dict.items():

        sorted_frames = sorted(gts_dict.keys())


        gt = [
            (gts_dict[frame].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            for frame in sorted_frames
        ]


        video_path = os.path.join(vid_root, f'{scene_name}_gt.mp4')


        save_video(gt, video_path, fps=fps)
        
# def save_flow_image(img, flo, save_path):
#     img = img[0].permute(1,2,0).cpu().numpy()
#     flo = flo[0].permute(1,2,0).cpu().numpy()
    
#     # map flow to rgb image
#     flo = flow_viz.flow_to_image(flo)
#     img_flo = np.concatenate([img, flo], axis=0)

#     # import matplotlib.pyplot as plt
#     # plt.imshow(img_flo / 255.0)
#     # plt.show()

#     # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
#     # cv2.waitKey()
    

#     cv2.imwrite(save_path, img_flo[:, :, [2,1,0]])