import torch
from model.diffusion_model import Unet, GaussianDiffusion
from data_loader.re10k_dataset import Re10k_dataset
from einops import rearrange
from PIL import Image
import numpy as np

if __name__ == '__main__':

    tensor = torch.randn(2, 2, 3)  # 举例：假设形状为 (b, c, w) = (2, 3, 4)
    threshold = 0.5

    print(tensor)

    tensor = tensor.permute(0,2,1)

    print(tensor)

