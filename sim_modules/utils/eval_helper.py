import torch
from d3gs.utils.loss_utils import l2_loss, ssim, psnr


def compute_eval_loss(renders_dict, gt_dict, pixel_loss):
    loss = 0.0
    length = len(renders_dict)
    for (renders_view, renders_frame_dict), (gt_view, gt_frame_dict) in zip(renders_dict.items(), gt_dict.items()):

        sorted_renders_frames = sorted(renders_frame_dict.keys())
        sorted_gt_frames = sorted(gt_frame_dict.keys())

        renders = [
            renders_frame_dict[frame]for frame in sorted_renders_frames
        ]
        gts = [
            gt_frame_dict[frame]for frame in sorted_gt_frames
        ]

        renders = torch.stack(renders, dim=0)
        gts = torch.stack(gts, dim=0)
        
        loss_rgb = pixel_loss(renders, gts)
        loss += loss_rgb
        
    return loss / length


def compute_eval_metrics(renders_dict, gt_dict):
    metrics = {}

    sorted_renders_frames = sorted(renders_dict.keys())
    sorted_gt_frames = sorted(gt_dict.keys())

    renders = [
        renders_dict[frame]for frame in sorted_renders_frames
    ]
    gts = [
        gt_dict[frame]for frame in sorted_gt_frames
    ]

    renders_BCHW = torch.stack(renders, dim=0)
    gts_BCHW = torch.stack(gts, dim=0)
    
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
        
        # if use_lpips:
        #     lpipss.append(lpips(render, gt, net_type='vgg'))
        
    metrics['rgb_loss'] = torch.tensor(rgb_loss).mean()
    metrics['psnr'] = torch.tensor(psnrs).mean()
    metrics['ssim'] = torch.tensor(ssims).mean()
    # if use_lpips:
    #     metrics['lpips'] = torch.tensor(lpipss).mean()
        
    return metrics