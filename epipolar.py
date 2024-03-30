import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from einops import einsum, rearrange, repeat
from data_loader.re10k_dataset import Re10k_dataset

import cv2

u = 51
v = 28

def get_epipolar_tensor(b,h,w,k,src_c2w,target_c2w):
    H = h
    W = H*16/9  #* 原始圖像為 16:9

    k = k.to(dtype=torch.float32)
    src_c2w=src_c2w.to(dtype=torch.float32)
    target_c2w=target_c2w.to(dtype=torch.float32)

    k = k.unsqueeze(0)
    src_c2w = src_c2w.unsqueeze(0)
    target_c2w = target_c2w.unsqueeze(0)

    #* unormalize intrinsic 

    k[:,0] = k[:,0]*W
    k[:,1] = k[:,1]*H

    k[:,0,2] = h/2
    k[:,1,2] = h/2

    device = k.device

    #* 創建 h*w 的 uv map
    x_coords, y_coords = torch.meshgrid(torch.arange(h), torch.arange(w))
    coords_tensor = torch.stack((x_coords.flatten(), y_coords.flatten(), torch.ones_like(x_coords).flatten()), dim=1)
    coords_tensor[:,[0,1]] = coords_tensor[:,[1,0]]
    coords_tensor = coords_tensor.to(dtype=torch.float32)
    coords_tensor = repeat(coords_tensor, 'HW p -> b HW p', b=b)

    x_coords = x_coords.to(device)
    y_coords = y_coords.to(device)
    coords_tensor = coords_tensor.to(device)

    k_3x3 = k[:,0:3,0:3]
    src_c2w_r = src_c2w[:,0:3,0:3]
    src_c2w_t = src_c2w[:,0:3,3]
    target_c2w_r = target_c2w[:,0:3,0:3]
    target_c2w_t = target_c2w[:,0:3,3]
    target_w2c_r = torch.linalg.inv(target_c2w_r)
    target_w2c_t = -target_c2w_t

    cx = k_3x3[:,0,2].view(b, 1)
    cy = k_3x3[:,1,2].view(b, 1)
    fx = k_3x3[:,0,0].view(b, 1)
    fy = k_3x3[:,1,1].view(b, 1)
    coords_tensor[...,0] = (coords_tensor[...,0]-cx)/fx
    coords_tensor[...,1] = (coords_tensor[...,1]-cy)/fy

    #* 做 H*W 個點的運算
    coords_tensor = rearrange(coords_tensor, 'b hw p -> b p hw') 
    point_3d_world = torch.matmul(src_c2w_r,coords_tensor)              #* 相機坐標系 -> 世界座標
    point_3d_world = point_3d_world + src_c2w_t.unsqueeze(-1)           #* 相機坐標系 -> 世界座標
    point_2d = torch.matmul(target_w2c_r,point_3d_world)                #* 世界座標 -> 相機座標
    point_2d = point_2d + target_w2c_t.unsqueeze(-1)                    #* 世界座標 -> 相機座標
    pi_to_j = torch.matmul(k_3x3,point_2d)                              #* 相機座標 -> 平面座標

    #* 原點的計算
    oi = torch.zeros(3).to(dtype=torch.float32)
    oi = repeat(oi, 'p -> b p', b=b)
    oi = oi.unsqueeze(-1)
    oi = oi.to(device)
    point_3d_world = torch.matmul(src_c2w_r,oi)
    point_3d_world = point_3d_world + src_c2w_t.unsqueeze(-1)  
    point_2d = torch.matmul(target_w2c_r,point_3d_world)
    point_2d = point_2d + target_w2c_t.unsqueeze(-1)  
    oi_to_j = torch.matmul(k_3x3,point_2d)
    oi_to_j = rearrange(oi_to_j, 'b c p -> b p c') #* (b,3,1) -> (b,1,3)

    #* 除以深度
    pi_to_j = rearrange(pi_to_j, 'b p hw -> b hw p') 
    pi_to_j = pi_to_j / pi_to_j[..., -1:]   #* (b,hw,3)
    oi_to_j = oi_to_j / oi_to_j[..., -1:]   #* (b,1,3)

    # print(f"pi_to_j: {pi_to_j[0,9]}")
    # print(f"oi_to_j: {oi_to_j[0,0]}")

    #* 計算feature map 每個點到每個 epipolar line 的距離
    coords_tensor = torch.stack((x_coords.flatten(), y_coords.flatten(), torch.ones_like(x_coords).flatten()), dim=1)
    coords_tensor[:,[0,1]] = coords_tensor[:,[1,0]]
    coords_tensor = coords_tensor.to(dtype=torch.float32) # (4096,3)
    coords_tensor = repeat(coords_tensor, 'HW p -> b HW p', b=b)
    coords_tensor = coords_tensor.to(device)

    oi_to_pi = pi_to_j - oi_to_j            #* h*w 個 epipolar line (b,hw,3)
    oi_to_coord = coords_tensor - oi_to_j   #* h*w 個點   (b,hw,3)

    ''''
        #* 這裡做擴展
        oi_to_pi    [0,0,0]     ->      oi_to_pi_repeat     [0,0,0]
                    [1,1,1]                                 [0,0,0]
                    [2,2,2]                                 [1,1,1]
                                                            [1,1,1]
                                                            .
                                                            .
                                                            .

        oi_to_coord     [0,0,0]     ->      oi_to_coord_repeat      [0,0,0]
                        [1,1,1]                                     [1,1,1]
                        [2,2,2]                                     [2,2,2]
                                                                    [0,0,0]
                                                                    .
                                                                    .
                                                                    .
    '''
    oi_to_pi_repeat = repeat(oi_to_pi, 'b i j -> b i (repeat j)',repeat = h*w)
    oi_to_pi_repeat = rearrange(oi_to_pi_repeat,"b i (repeat j) -> b (i repeat) j", repeat = h*w)
    oi_to_coord_repeat = repeat(oi_to_coord, 'b i j -> b (repeat i) j',repeat = h*w)


    area = torch.cross(oi_to_pi_repeat,oi_to_coord_repeat,dim=-1)     #* (b,hw*hw,3)
    area = torch.norm(area,dim=-1 ,p=2)
    vector_len = torch.norm(oi_to_pi_repeat, dim=-1, p=2)
    distance = area/vector_len
    distance = 1 - torch.sigmoid(50*(distance-0.5))

    epipolar_map = rearrange(distance,"b (hw hw2) -> b hw hw2",hw = h*w)

    test_map = epipolar_map[0][v*h+u]
    test_map = rearrange(test_map, '(h w) -> h w',h=64)
    test_map = (test_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite('test_folder/target_epipolar_atten.png', test_map)

    return epipolar_map
        

def get_epipolar(src_img,target_img,k,src_c2w,target_c2w):
    c,h,w = src_img.shape

    H = h
    W = H*16/9  #* 原始圖像為 16:9

    #* unormalize intrinsic 

    k[0] = k[0]*W
    k[1] = k[1]*H

    k[0,2] = h/2
    k[1,2] = h/2

    print(f"k = {k}")

    # u = 25
    # v = 30

    k_3x3 = k[0:3,0:3]
    src_c2w_r = src_c2w[0:3,0:3]
    src_c2w_t = src_c2w[0:3,3]
    target_c2w_r = target_c2w[0:3,0:3]
    target_c2w_t = target_c2w[0:3,3]

    src_c2w_r = src_c2w_r.numpy() 
    src_c2w_t = src_c2w_t.numpy()       
    k_3x3 = k_3x3.numpy()               
    target_w2c_r = np.linalg.inv(target_c2w_r)  #* world_to_camera -> camera_to_world
    target_w2c_t = -target_c2w_t.numpy()

    cx = k_3x3[0,2]
    cy = k_3x3[1,2]
    fx = k_3x3[0,0]
    fy = k_3x3[1,1]
    u_plus = (u-cx)/fx
    v_plus = (v-cy)/fy

    pi = np.array([u_plus,v_plus,1])
    oi = np.array([0,0,0])
    print(f"pi:{pi}")

    point_3d_camera = pi                                          #* 平面座標 -> 相機坐標系的3D點
    point_3d_world = src_c2w_r.dot(point_3d_camera) + src_c2w_t   #* 相機坐標系 -> 世界座標
    # print(f"point_3d_world:{point_3d_world}")
    point_2d = target_w2c_r.dot(point_3d_world) + target_w2c_t    #* 世界座標 -> 相機座標
    # print(f"point_2d:{point_2d}")
    pi_to_j = k_3x3.dot(point_2d[:3])                             #* 相機座標 -> 平面座標

    point_3d_camera = oi                                          #* 平面座標 -> 相機坐標系的3D點
    point_3d_world = src_c2w_r.dot(point_3d_camera) + src_c2w_t   #* 相機坐標系 -> 世界座標
    point_2d = target_w2c_r.dot(point_3d_world) + target_w2c_t    #* 世界座標 -> 相機座標
    oi_to_j = k_3x3.dot(point_2d[:3])                             #* 相機座標 -> 平面座標

    print(f"point_3d_world:{point_3d_world}")
    print(f"point_2d:{point_2d}")
    print(f"oi_to_j before:{oi_to_j}")

    pi_to_j /= pi_to_j[2]   #* 除深度
    oi_to_j /= oi_to_j[2]

    pi_to_j = pi_to_j.astype(np.int32)
    oi_to_j = oi_to_j.astype(np.int32)

    print(f"pi_to_j: {pi_to_j}") # (u,v) 0 9 -> (x,y) 15 13
    print(f"oi_to_j: {oi_to_j}")

    print(f"uv = ({u},{v})")

    point1 = (pi_to_j[0],pi_to_j[1])
    point2 = (oi_to_j[0],oi_to_j[1])

    m = (point2[1]-point1[1]) / (point2[0]-point1[0])
    b = point1[1] - m*point1[0]

    x_axis = np.arange(h)
    y_value = m*x_axis + b
    match_coor = np.where((y_value >= 0) & (y_value < h))
    min_match = match_coor[0].min()
    max_match = match_coor[0].max()

    point1 = (min_match,int(min_match*m+b))
    point2 = (max_match,int(max_match*m+b))

    color = (0, 0, 255)
    thickness = 2
    target_img = (target_img+1)/2
    target_img = (target_img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    target_img = np.ascontiguousarray(target_img)
    cv2.line(target_img, point1, point2, color, thickness)

    point = (u,v)
    src_img = (src_img+1)/2
    src_img = (src_img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    src_img = np.ascontiguousarray(src_img)
    cv2.circle(src_img, point, radius=3, color=color, thickness=-1)

    cv2.imwrite('test_folder/target_epipolar.png', target_img)
    cv2.imwrite('test_folder/src_point.png', src_img)

    return 1

if __name__ == '__main__':

    from midas.model_loader import default_models, load_model
    model_type = "dpt_swin2_tiny_256"
    model_weights = default_models[model_type]
    midas_model, midas_transform, net_w, net_h = load_model("cpu", model_weights, model_type, False, None, False)

    test = Re10k_dataset("../../../disk2/icchiu","train",midas_transform = midas_transform)
    data = test[21]
    # print(data['img'].shape)
    # print(data['intrinsics'])
    # print(data['c2w'][0])
    # print(data['c2w'][1])

    prev_img = data['img'][0]
    now_img = data['img'][1]
    k = data['intrinsics']
    k2 = k.clone()
    prev_c2w = data['c2w'][0]
    now_c2w = data['c2w'][1]

    get_epipolar(prev_img,now_img,k,prev_c2w,now_c2w)

    device = "cuda"
    k2 = k2.to(device)
    prev_c2w = prev_c2w.to(device)
    now_c2w = now_c2w.to(device)
    get_epipolar_tensor(1,64,64,k2,prev_c2w,now_c2w)

    