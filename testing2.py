import torch
from model.diffusion_model import Unet, GaussianDiffusion
from data_loader.re10k_dataset import Re10k_dataset
from einops import rearrange
from PIL import Image
import numpy as np

if __name__ == '__main__':

    # 創建一個 (h, w) 的 NumPy 數組
    array = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1]])

    # 使用 numpy.where 找到數組中等於 1 的索引
    indices = np.where(array == 1)

    print(indices)
