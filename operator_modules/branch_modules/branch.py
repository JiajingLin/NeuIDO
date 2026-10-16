import os
import torch
import torch.nn as nn
from omegaconf import DictConfig
import mpu
from .vae.autoencoder import VideoAutoencoderInferenceWrapper
from .vae.cp_enc_dec import ContextParallelEncoder3D
from .vae.utils import instantiate_from_config

class VideoBranchEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        initialize_distributed(device=f'cuda:{cfg.gpu}')
        # load vae model
        # load vae model
        vae_model: VideoAutoencoderInferenceWrapper = None
        vae_model = instantiate_from_config(cfg.operator.branch.vae)
        vae_model.requires_grad_(False)
        vae_model.eval()
        vae_model.bfloat16()
        

        self.flow_encoder: ContextParallelEncoder3D = vae_model.encoder
        # self.flow_encoder.requires_grad_(False)
        # self.flow_encoder.eval()
        
        # ---- hyperparams (with safe defaults) ----
        out_dim = getattr(cfg.operator, "latent_dim", 128)  # latent_dim
        
        c_in = getattr(cfg.operator.branch, "vae_latent_channels", 4)
        c_mid = getattr(cfg.operator.branch, "conv_mid_channels", 128)
        num_conv_layers = getattr(cfg.operator.branch, "conv_layers", 2)
        attn_heads = getattr(cfg.operator.branch, "attn_heads", 4)
        mlp_hidden = getattr(cfg.operator.branch, "mlp_hidden", 256)
        dropout = getattr(cfg.operator.branch, "dropout", 0.0)
        
        # ---- 3D Conv stack (lightweight) ----
        convs = []
        c_prev = c_in
        for i in range(num_conv_layers):
            convs += [
                nn.Conv3d(c_prev, c_mid, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=c_mid),
                nn.SiLU()
            ]
            c_prev = c_mid
        self.conv3d = nn.Sequential(*convs)

        # ---- projection (optional but recommended) ----
        # already c_mid, so projection is identity-ish; keep a linear to be safe
        self.token_proj = nn.Linear(c_mid, c_mid)

        # ---- learnable pooling query + multihead attention ----
        # one query per batch, shape: (1, 1, c_mid)
        self.pool_q = nn.Parameter(torch.randn(1, 1, c_mid))

        self.pool_attn = nn.MultiheadAttention(
            embed_dim=c_mid,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True
        )

        # ---- final MLP to out_dim ----
        self.elasticity_mlp = nn.Sequential(
            nn.Linear(c_mid, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_dim)
        )
        
        self.plasticity_mlp = nn.Sequential(
            nn.Linear(c_mid, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_dim)
        )
        
        self.latent_cache = {}
        


    def forward(self, x: torch.Tensor, motion_ids: torch.Tensor = None) -> torch.Tensor:
        """
        x: input flow/trajectory-visual video to VAE encoder
        motion_ids: motion ids to correspond to the flow video, shape (b,) 默认为None，表示需要进行完整的前向
        return: global motion vector (b, out_dim)
        """
        # VAE latent: (b, c, l, h, w,)

        with torch.no_grad():
            if motion_ids is None:
                z = self.flow_encoder(x)
            else:

                l = []
                for motion_id, x_i in zip(motion_ids, x):
                    key = motion_id.item()
                    if key not in self.latent_cache:
                        self.latent_cache[key] = self.flow_encoder(x_i.unsqueeze(0))
                        torch.cuda.empty_cache()
                    l.append(self.latent_cache[key]) # (b, c_mid, l, h, w)
                z = torch.cat(l, dim=0) # (b, c_mid, l, h, w)
            

        z = z.to(torch.float32)
        
        # print(z.mean())
        # print(z.max())
        # print(z.min())

        # Conv3D wants (b, c, l, h, w)
        # z = z.permute(0, 4, 1, 2, 3).contiguous()  # (b, c, l, h, w)
        z = self.conv3d(z)                         # (b, c_mid, l, h, w)
        
        # print(z.mean())
        # print(z.max())
        # print(z.min())

        # flatten to tokens: (b, n, c_mid)
        b, c_mid, l, h, w = z.shape
        z = z.permute(0, 2, 3, 4, 1).contiguous()  # (b, l, h, w, c_mid)
        tokens = z.view(b, l * h * w, c_mid)       # (b, n, c_mid)
        tokens = self.token_proj(tokens)

        # print(tokens.mean())
        # print(tokens.max())
        # print(tokens.min())

        # attention pooling
        # pool_q expand to batch: (b, 1, c_mid)
        q = self.pool_q.expand(b, -1, -1)
        
        # print(q)

        # MultiheadAttention(query, key, value)
        pooled, _ = self.pool_attn(q, tokens, tokens)  # (b, 1, c_mid)
        pooled = pooled.squeeze(1)                     # (b, c_mid)
        
        # print(pooled.mean())
        # print(pooled.max())
        # print(pooled.min())
        
        # map to out_dim
        elasticity_out = self.elasticity_mlp(pooled)                         # (b, out_dim)
        plasticity_out = self.plasticity_mlp(pooled)                         # (b, out_dim)
        
        # print(out.mean())
        # print(out.max())
        # print(out.min())
        
        return elasticity_out, plasticity_out




def initialize_distributed(device='cuda:0'):
    from datetime import timedelta
    import warnings
    from operator_modules.branch_modules.vae.context_parallel_util import initialize_context_parallel, set_context_parallel_group
    
    distributed_backend = 'nccl'
    world_size = 1
    rank = 0
    model_parallel_size = 1
    is_deepspeed = False
    
    """Initialize torch.distributed."""
    if torch.distributed.is_initialized():
        if mpu.model_parallel_is_initialized():
            if model_parallel_size != mpu.get_model_parallel_world_size():
                raise ValueError(
                    "model_parallel_size is inconsistent with prior configuration."
                    "We currently do not support changing model_parallel_size."
                )
            return False
        else:
            if model_parallel_size > 1:
                warnings.warn(
                    "model_parallel_size > 1 but torch.distributed is not initialized via SAT."
                    "Please carefully make sure the correctness on your own."
                )
            mpu.initialize_model_parallel(model_parallel_size)
        return True
    # the automatic assignment of devices has been moved to arguments.py
    if device == "cpu":
        pass
    else:
        torch.cuda.set_device(device)
    # Call the init process
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")

    if world_size == 1:
        from mpu.helpers import get_free_port
        default_master_port = str(get_free_port())
    else:
        default_master_port = "6000"
    master_port = os.getenv("MASTER_PORT", default_master_port)
    init_method += master_ip + ":" + master_port
    torch.distributed.init_process_group(
        backend=distributed_backend,
        world_size=world_size,
        rank=rank,
        init_method=init_method,
        timeout=timedelta(seconds=7200000),
    )

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(model_parallel_size)

    # Set vae context parallel group equal to model parallel group

    if model_parallel_size <= 2:
        set_context_parallel_group(model_parallel_size, mpu.get_model_parallel_group())
    else:
        initialize_context_parallel(2)
    # mpu.initialize_model_parallel(1)
    # Optional DeepSpeed Activation Checkpointing Features
    if is_deepspeed:
        import deepspeed

        deepspeed.init_distributed(
            dist_backend=distributed_backend, world_size=world_size, rank=rank, init_method=init_method
        )
        # # It seems that it has no negative influence to configure it even without using checkpointing.
        # deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    else:
        # in model-only mode, we don't want to init deepspeed, but we still need to init the rng tracker for model_parallel, just because we save the seed by default when dropout.
        try:
            import deepspeed
            from deepspeed.runtime.activation_checkpointing.checkpointing import (
                _CUDA_RNG_STATE_TRACKER,
                _MODEL_PARALLEL_RNG_TRACKER_NAME,
            )

            _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 1)  # default seed 1
        except Exception as e:
            from mpu.helpers import print_rank0
            print_rank0(str(e), level="DEBUG")

    return True