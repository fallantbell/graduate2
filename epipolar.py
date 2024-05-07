import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from einops import rearrange, repeat
from data_loader.re10k_dataset import Re10k_dataset
import torch.nn.functional as F
import torchvision.transforms as T

import cv2

u = 50
v = 118

index = 0

steep = 50
error_range = 0.5

def normalize(weight):
    min_val = weight.min()
    max_val = weight.max()
    weight = (weight - min_val) / (max_val - min_val)

    return weight

def get_epipolar_tensor(b,h,w,k,src_c2w,target_c2w):
    H = h
    W = H*16/9  #* 原始圖像為 16:9

    k = k.to(dtype=torch.float32)
    src_c2w=src_c2w.to(dtype=torch.float32)
    target_c2w=target_c2w.to(dtype=torch.float32)

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
    pi_to_j_unnormalize = rearrange(pi_to_j, 'b p hw -> b hw p') 
    pi_to_j = pi_to_j_unnormalize / (pi_to_j_unnormalize[..., -1:] + 1e-6)   #* (b,hw,3)
    # pi_to_j = pi_to_j_unnormalize / pi_to_j_unnormalize[..., -1:]
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

    distance_weight = 1 - torch.sigmoid(steep*(distance-error_range)) # 50 0.5
    # distance_weight = 1 - torch.sigmoid(steep*(distance-0.05*H)) # 50 0.5

    epipolar_map = rearrange(distance_weight,"b (hw hw2) -> b hw hw2",hw = h*w)

    #* 如果 max(1-sigmoid) < 0.5 
    #* => min(distance) > 0.05 
    #* => 每個點離epipolar line 太遠
    #* => epipolar line 不在圖中
    #* weight map 全設為 1 
    max_values, _ = torch.max(epipolar_map, dim=-1)
    mask = max_values < 0.5
    epipolar_map[mask.unsqueeze(-1).expand_as(epipolar_map)] = 1

    # test_map = epipolar_map[0][v*h+u]
    # test_map = rearrange(test_map, '(h w) -> h w',h=128)
    # test_map = (test_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # cv2.imwrite('test_folder/target_epipolar_atten.png', test_map)

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

    # color = (0, 0, 255)
    # thickness = 2
    # target_img = (target_img+1)/2
    # target_img = (target_img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    # target_img = np.ascontiguousarray(target_img)
    # cv2.line(target_img, point1, point2, color, thickness)

    # point = (u,v)
    # src_img = (src_img+1)/2
    # src_img = (src_img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    # src_img = np.ascontiguousarray(src_img)
    # cv2.circle(src_img, point, radius=3, color=color, thickness=-1)

    # img = Image.fromarray(np.uint8(target_img))
    # img.save(f'test_folder/target_epipolar{index}.png')
    # img = Image.fromarray(np.uint8(src_img))
    # img.save(f'test_folder/src_point.png')

    return point1, point2

if __name__ == '__main__':

    from midas.model_loader import default_models, load_model
    model_type = "dpt_swin2_tiny_256"
    model_weights = default_models[model_type]
    midas_model, midas_transform, net_w, net_h = load_model("cpu", model_weights, model_type, False, None, False)

    test = Re10k_dataset("../dataset","test",midas_transform = midas_transform,do_latent = False)
    data = test[0] # 150

    prev_img = data['img'][0]
    now_img = data['img'][1]
    k = data['intrinsics']
    k2 = k.clone()
    prev_c2w = data['c2w'][0]
    now_c2w = data['c2w'][1]

    # points = []
    # color_list = [(84,255,159),(0,255,255),(255,255,0),(255,165,0),(4, 48, 255)]
    # point_list = [(125,72),(124,76),(107,70),(17,51),(4,48)]
    # for (u2,v2) in point_list:
    #     u = u2
    #     v = v2
    #     prev_img = data['img'][0]
    #     now_img = data['img'][1]
    #     k = data['intrinsics']
    #     k2 = k.clone()
    #     prev_c2w = data['c2w'][0]
    #     now_c2w = data['c2w'][1]
    
    #     p1,p2 = get_epipolar(now_img,prev_img,k2,prev_c2w,now_c2w)

    #     points.append((p1,p2))
    
    # total_img = (now_img+1)/2
    # total_img = (total_img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    # total_img = np.ascontiguousarray(total_img)

    # for i in range(5):
    #     target_img = (now_img+1)/2
    #     target_img = (target_img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    #     target_img = np.ascontiguousarray(target_img)
    #     color = color_list[i]
    #     thickness = 1
    #     cv2.line(target_img, points[i][0], points[i][1], color, thickness)
    #     cv2.line(total_img, points[i][0], points[i][1], color, thickness)

    #     cv2.line(target_img,(0,60),(60-1,60),(255,255,255),1)
    #     cv2.line(target_img,(60,0),(60,60-1),(255,255,255),1)

    #     img = Image.fromarray(np.uint8(target_img))
    #     img.save(f'test_folder/back_epipolars_{i}.png')
    # img = Image.fromarray(np.uint8(total_img))
    # img.save(f'test_folder/back_epipolars_all.png')

    # sys.exit()
    h = 128

    device = "cuda"
    k2 = k.clone()
    k2 = k2.unsqueeze(0).to(device)
    prev_c2w = prev_c2w.unsqueeze(0).to(device)
    now_c2w = now_c2w.unsqueeze(0).to(device)
    forward_epipolar_map = get_epipolar_tensor(1,h,h,k2.clone(),now_c2w,prev_c2w)
    forward_epipolar_map = forward_epipolar_map[0]

    backward_epipolar_map_ori = get_epipolar_tensor(1,h,h,k2.clone(),prev_c2w,now_c2w)
    backward_epipolar_map_transpose = backward_epipolar_map_ori.permute(0,2,1)
    backward_epipolar_map = backward_epipolar_map_transpose[0]
    backward_epipolar_map_ori = backward_epipolar_map_ori[0]

    weight_map = torch.ones_like(forward_epipolar_map)
    weight_map = weight_map
    weight_map_forward = weight_map * forward_epipolar_map          #* forward epipolar
    weight_map_backward = weight_map * backward_epipolar_map        #* backward epipolar transpose
    weight_map_backward_ori = weight_map * backward_epipolar_map_ori    #*  backward epipolar
    weight_map_bidirection = weight_map_forward * backward_epipolar_map    

    # weight_map_forward = weight_map_forward.softmax(dim=-1)
    # weight_map_bidirection = weight_map_bidirection.softmax(dim=-1)

    foldername = f"test_folder_steep{steep}_error{error_range}_test"
    os.makedirs(f"{foldername}",exist_ok=True)
    for i in range(10):
        if i!=5:
            continue
        u = np.random.randint(h)
        v = np.random.randint(h)
        u = (h//10)*i
        v = u
        os.makedirs(f"{foldername}/u_{u}_v_{v}",exist_ok=True)

        # src point image
        color = (0, 0, 255)
        src_img = now_img
        point = (u,v)
        src_img = (src_img+1)/2
        src_img = (src_img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        src_img = np.ascontiguousarray(src_img)
        # cv2.circle(src_img, point, radius=2, color=color, thickness=-1)
        cv2.line(src_img,(0,u),(v-1,u),(255,255,255),1)
        cv2.line(src_img,(v,0),(v,u-1),(255,255,255),1)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/src_point.png', src_img)

        # target image
        target_img = prev_img
        target_img = (target_img+1)/2
        target_img = (target_img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        target_img = np.ascontiguousarray(target_img)

        #* forward
        weight = weight_map_forward[v*h+u]
        weight = rearrange(weight, '(h w) -> h w',h=h)
        weight = normalize(weight)
        weight = (weight.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/forward_epipolar_map.png', weight)

        weight = cv2.merge([weight,weight,weight])
        forward_result = cv2.addWeighted(target_img, 0.5, weight, 0.5, 0)
        forward_result = cv2.cvtColor(forward_result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/forward_epipolar.png',forward_result)

        #* backward
        weight = weight_map_backward[v*h+u]
        weight = rearrange(weight, '(h w) -> h w',h=h)
        weight = normalize(weight)
        weight = (weight.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        #* 找backward epipolar map 中空白區域，隨機取幾個點
        white_area = np.where(weight)
        point_num = len(white_area)
        index = np.random.randint(point_num)
        p0 = (white_area[0][index],white_area[1][index])
        index = np.random.randint(point_num)
        p1 = (white_area[0][index],white_area[1][index])
        index = np.random.randint(point_num)
        p2 = (white_area[0][index],white_area[1][index])

        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/backward_epipolar_map.png', weight)

        weight = cv2.merge([weight,weight,weight])
        backward_result = cv2.addWeighted(target_img, 0.5, weight, 0.5, 0)
        cv2.circle(backward_result, p0, radius=2, color=(0,255,255), thickness=-1)
        cv2.circle(backward_result, p1, radius=2, color=(255,0,255), thickness=-1)
        cv2.circle(backward_result, p2, radius=2, color=(255,255,0), thickness=-1)

        backward_result = cv2.cvtColor(backward_result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/backward_epipolar.png',backward_result)

        #* 將這幾個點反推回 target img 的epipolar line，看有沒有經過給定的點
        color_list = [(0,255,255),(255,0,255),(255,255,0)]
        point_list = [p0,p1,p2]
        total_img = (now_img+1)/2
        total_img = (total_img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        total_img = np.ascontiguousarray(total_img)
        idx = 0
        for (u2,v2) in point_list:
            u = u2
            v = v2
            prev_img = data['img'][0]
            now_img = data['img'][1]
            k = data['intrinsics']
            k2 = k.clone()
            prev_c2w = data['c2w'][0]
            now_c2w = data['c2w'][1]
        
            p1,p2 = get_epipolar(now_img,prev_img,k2,prev_c2w,now_c2w)
            
            target_img = (now_img+1)/2
            target_img = (target_img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            target_img = np.ascontiguousarray(target_img)
            color = color_list[idx]
            idx+=1
            thickness = 1
            cv2.line(target_img, p1, p2, color, thickness)
            cv2.line(total_img, p1, p2, color, thickness)

            cv2.line(target_img,(0,u),(v-1,u),(255,255,255),1)
            cv2.line(target_img,(v,0),(v,u-1),(255,255,255),1)

            img = Image.fromarray(np.uint8(target_img))
            img.save(f'{foldername}/back_epipolars_{u2}_{v2}.png')
        
        img = Image.fromarray(np.uint8(total_img))
        img.save(f'{foldername}/back_epipolars_all.png')

        #* bidirection
        weight_map = weight_map_bidirection[v*h+u]
        weight = rearrange(weight_map, '(h w) -> h w',h=h)
        weight = normalize(weight)
        weight = (weight.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/bidirection_epipolar_map.png', weight)

        weight = cv2.merge([weight,weight,weight])
        bidirection_result = cv2.addWeighted(target_img, 0.5, weight, 0.5, 0)
        bidirection_result = cv2.cvtColor(bidirection_result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/bidirection_epipolar.png',bidirection_result)

        #* guassion blur 測試

        k = 3
        weight = rearrange(weight_map, '(h w) -> h w',h=h)

        # kernel_tensor = torch.rand(k,k).to(dtype=torch.float32)
        # kernel_tensor = kernel_tensor.to(device)
        # blurred_tensor = F.conv2d(weight.unsqueeze(0).unsqueeze(0), \
        #                           kernel_tensor.unsqueeze(0).unsqueeze(0), padding=(k-1)//2)
        # blurred_tensor = blurred_tensor.squeeze()
        transform1 = T.GaussianBlur(k,1.5)
        blurred_tensor = transform1(weight.unsqueeze(0).unsqueeze(0))
        blurred_tensor = blurred_tensor.squeeze()

        weight = blurred_tensor
        weight = normalize(weight)
        weight = (weight.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/bidirection_epipolar_map_BLUR{k}.png', weight)
        weight = cv2.merge([weight,weight,weight])
        bidirection_result = cv2.addWeighted(target_img, 0.5, weight, 0.5, 0)
        bidirection_result = cv2.cvtColor(bidirection_result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/bidirection_epipolar_BLUR{k}.png',bidirection_result)

        k = 5
        weight = rearrange(weight_map, '(h w) -> h w',h=h)

        # kernel_tensor = torch.rand(k,k).to(dtype=torch.float32)
        # kernel_tensor = kernel_tensor.to(device)
        # blurred_tensor = F.conv2d(weight.unsqueeze(0).unsqueeze(0), \
        #                           kernel_tensor.unsqueeze(0).unsqueeze(0), padding=(k-1)//2)
        # blurred_tensor = blurred_tensor.squeeze()
        transform1 = T.GaussianBlur(k,1.5)
        blurred_tensor = transform1(weight.unsqueeze(0).unsqueeze(0))
        blurred_tensor = blurred_tensor.squeeze()

        weight = blurred_tensor
        weight = normalize(weight)
        weight = (weight.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/bidirection_epipolar_map_BLUR{k}.png', weight)
        weight = cv2.merge([weight,weight,weight])
        bidirection_result = cv2.addWeighted(target_img, 0.5, weight, 0.5, 0)
        bidirection_result = cv2.cvtColor(bidirection_result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/bidirection_epipolar_BLUR{k}.png',bidirection_result)

        k = 7
        weight = rearrange(weight_map, '(h w) -> h w',h=h)

        # kernel_tensor = torch.rand(k,k).to(dtype=torch.float32)
        # kernel_tensor = kernel_tensor.to(device)
        # blurred_tensor = F.conv2d(weight.unsqueeze(0).unsqueeze(0), \
        #                           kernel_tensor.unsqueeze(0).unsqueeze(0), padding=(k-1)//2)
        # blurred_tensor = blurred_tensor.squeeze()
        transform1 = T.GaussianBlur(k,1.5)
        blurred_tensor = transform1(weight.unsqueeze(0).unsqueeze(0))
        blurred_tensor = blurred_tensor.squeeze()

        weight = blurred_tensor
        weight = normalize(weight)
        weight = (weight.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/bidirection_epipolar_map_BLUR{k}.png', weight)
        weight = cv2.merge([weight,weight,weight])
        bidirection_result = cv2.addWeighted(target_img, 0.5, weight, 0.5, 0)
        bidirection_result = cv2.cvtColor(bidirection_result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{foldername}/u_{u}_v_{v}/bidirection_epipolar_BLUR{k}.png',bidirection_result)