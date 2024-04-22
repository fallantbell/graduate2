import argparse
import os
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from einops import rearrange
import numpy as np
from utils import prepare_device
from utils import read_json, write_json

import cv2

from PIL import Image

from midas.model_loader import default_models, load_model

def init_obj(config,name, module, *args, **kwargs):
    module_name = config[name]['type']
    module_args = dict(config[name]['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)

def main(config):

    #* 宣告midas model
    model_type = "dpt_swin2_tiny_256"
    model_weights = default_models[model_type]
    midas_model, midas_transform, net_w, net_h = load_model("cpu", model_weights, model_type, False, None, False)

    infer_len = config['args']['infer_len']
    max_interval = config['args']['max_interval']
    batch_size = config['data_loader']['args']['batch_size']
    same_start = config['args']['same_start']
    do_epipolar = config['args']['do_epipolar']
    do_mae = config['args']['do_mae']

    video_name = config['name']
    video_name += f"_epoch{config['resume'][-7:-4]}"
    
    video_name += f"_inter{max_interval}"

    if same_start:
        video_name += "_samestart"
    else:
        video_name += "_randstart"
    
    print(f"video_name = {video_name}")

    #* 宣告 test dataloader
    test_data_loader = init_obj(config,'data_loader', module_data, 
                                        mode="test",midas_transform = midas_transform,
                                        infer_len = infer_len, 
                                        )

    #* 宣告model
    from model.diffusion_model import Unet, GaussianDiffusion

    unet_model = Unet(
        dim = 128,
        init_dim = 128,
        dim_mults = (2, 4, 8),
        channels=3, 
        out_dim=3,
        do_epipolar = do_epipolar,
        do_mae = do_mae,
    )
    
    model = GaussianDiffusion(
        unet_model,
        timesteps = 1000,    # number of steps
        sampling_timesteps = 50,  # ddim sample
        beta_schedule = 'cosine',
    )

    #* 放入 gpu
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    midas_model = midas_model.to(device)
    # if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    #* 讀取checkpoint
    checkpoint = torch.load(config['resume'])
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    model.eval()

    #* inference
    total_video = []
    gt_total_video = []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_data_loader)):
            #* img shape (b,infer_num,3,h,w)
            #* intrinsic 是 (b,4,4) 
            #* c2w shape (b,infer_num,4,4)

            img = data['img']
            intrinsic = data['intrinsics'].to(device)
            c2w = data['c2w'].to(device)

            #* 將img tensor 還原成 (0,1) 的圖像形式
            img = rearrange(img, 'b infer c h w -> b infer h w c')
            img = (img+1)/2
            # img = img.numpy()

            output_video = []
            output_video.append(img[:,0]) #* 放入起始image


            shape = torch.randn(batch_size, 3,64,64).shape
            xT_noise = torch.randn(shape, device = device)

            for i in range(1,infer_len): #* 跑 infer_len 張圖片

                #* 隨機取過去的影像當作 condition
                next_frame = i
                interval_len = np.random.randint(max_interval) + 1
                prev_frame = max(next_frame-interval_len,0)
                # prev_frame = 0

                #* 過去和要預測影像的 extrinsic
                new_c2w = []
                new_c2w.append(c2w[:,prev_frame])
                new_c2w.append(c2w[:,next_frame])
                new_c2w = torch.stack(new_c2w,dim=1)

                #* 對過去的影像經過 midas transform 再做 midas
                src_img_tensor = []
                src_img_numpy = output_video[prev_frame].numpy() #* (b,h,w,c)
                for j in range(batch_size):
                    src_img = midas_transform({"image": src_img_numpy[j]})["image"]
                    src_img = torch.from_numpy(src_img)
                    src_img_tensor.append(src_img)
                src_img_tensor = torch.stack(src_img_tensor,dim=0)
                src_img_tensor = src_img_tensor.to(device)

                src_l2, src_l3,src_l4 = midas_model.forward(src_img_tensor)

                if same_start != True:
                    shape = torch.randn(batch_size, 3,64,64).shape
                    xT_noise = torch.randn(shape, device = device)
                    
                #* output 為下一張影像，shape (b,c,h,w), (0,1)  
                output = model.module.ddim_sample(shape,xT_noise, src_l2, src_l3, src_l4, K = intrinsic, c2w = new_c2w)
                
                output = rearrange(output,'b c h w -> b h w c')
                output_video.append(output.cpu())
            
            output_video_tensor = torch.stack(output_video,dim=1)
            output_video_numpy = (output_video_tensor.clamp(0, 1).numpy() * 255).astype(np.uint8)
            gt_img_numpy = (img.clamp(0, 1).numpy() * 255).astype(np.uint8)

            for i in range(output_video_numpy.shape[0]):
                one_video = []
                gt_one_video = []
                for j in range(output_video_numpy.shape[1]):
                    one_video.append(output_video_numpy[i][j])
                    gt_one_video.append(gt_img_numpy[i,j])

                total_video.append(one_video)
                gt_total_video.append(gt_one_video)

            if (batch_idx+1)*batch_size > 0:
                break

    for i in range(len(total_video)):
        video_dir = f"saved_video/{video_name}/{i}"
        gt_video_dir = f"saved_video/gt/{i}"
        os.makedirs(video_dir,exist_ok=True)
        os.makedirs(gt_video_dir,exist_ok=True)
        for j in range(len(total_video[i])):
            # cv2.imwrite(f'{video_dir}/{j}.png', total_video[i][j])
            # cv2.imwrite(f'{gt_video_dir}/{j}.png', gt_total_video[i][j])
            img = Image.fromarray(np.uint8(total_video[i][j]))
            img.save(f'{video_dir}/{j}.png')

            img = Image.fromarray(np.uint8(gt_total_video[i][j]))
            img.save(f'{gt_video_dir}/{j}.png')

    print(f"save video len {infer_len} at saved_video/{video_name}")
    print(f"save gt video len {infer_len} at saved_video/gt")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    args = args.parse_args()
    
    # config = ConfigParser.from_args(args)
    config = read_json(args.config)
    config['resume'] = args.resume
    main(config)
