import torch
import os
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from einops import rearrange
from data_loader.dataset import ACIDdataset


def get_epipolar(src_img,target_img,k,src_w2c,target_w2c):
    c,h,w = src_img.shape

    u = random.randint(0,h-1)
    v = random.randint(0,w-1)

    k = k[0:3,0:3]
    src_w2c = src_w2c[0:3]
    target_w2c = target_w2c[0:3]

    print(f"w2c shape = {src_w2c.shape}")
    print(f"k shape = {k.shape}")

    src_c2w = np.linalg.pinv(src_w2c)
    k_inv = np.linalg.inv(k)
    target_w2c = target_w2c.numpy()

    pi = np.array([u,v,1])
    oi = np.array([0,0,0])

    print(f"c2w shape = {src_c2w.shape}")
    print(f"w2c shape = {target_w2c.shape}")
    print(f"k_inv shape = {k_inv.shape}")
    print(f"pi shape = {pi.shape}")

    point_2d_normalized = k_inv.dot(pi)
    point_3d_camera = src_c2w.dot(point_2d_normalized)
    point_2d = target_w2c.dot(point_3d_camera)
    pi_to_j = k.dot(point_2d)

    point_2d_normalized = k_inv.dot(oi)
    point_3d_camera = src_c2w.dot(point_2d_normalized)
    point_2d = target_w2c.dot(point_3d_camera)
    oi_to_j = k.dot(point_2d)

    print(f"pi_to_j: {pi_to_j}")
    print(f"oi_to_j: {oi_to_j}")

    return 1

if __name__ == '__main__':
    test = ACIDdataset("../../../../disk2/icchiu","train")
    data = test[0]
    print(data['img'].shape)
    print(data['intrinsics'])
    print(data['w2c'][0])

    prev_img = data['img'][0]
    now_img = data['img'][1]
    k = data['intrinsics']
    prev_w2c = data['w2c'][0]
    now_w2c = data['w2c'][1]

    get_epipolar(prev_img,now_img,k,prev_w2c,now_w2c)

    