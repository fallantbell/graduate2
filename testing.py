import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch

from midas.model_loader import default_models, load_model

from model.diffusion_model import Unet, GaussianDiffusion


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total num = {total_num}")
    print(f"trainable num = {trainable_num}")

model_type = "dpt_swin2_tiny_256"
model_weights = default_models[model_type]
midas_model, midas_transform, net_w, net_h = load_model("cpu", model_weights, model_type, False, None, False)

unet_model = Unet(
    dim = 128,
    init_dim = 128,
    dim_mults = (2, 4, 8),
    channels=3, 
    out_dim=3,
    do_epipolar = True,
    do_mae= True,
)

model = GaussianDiffusion(
    unet_model,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 50,  # ddim sample
    beta_schedule = 'cosine',
)

get_parameter_number(model)



