from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import lpips

default_transform = transforms.Compose([
    transforms.ToTensor(),
])

device = "cuda"

loss_fn_alex = lpips.LPIPS(net='alex').to(device)

def calc_ssim(img1, img2):
    '''
        img1,img2: PIL Image
    '''

    img1, img2 = np.array(img1), np.array(img2)

    ssim_score = ssim(img1, img2, channel_axis = 2)

    return ssim_score

def calc_psnr(img1, img2):
    '''
        img1,img2: PIL Image
    '''
    img1, img2 = np.array(img1), np.array(img2)
    psnr_score = psnr(img1, img2, data_range=255)

    return psnr_score

def calc_lpips(img1_tensor, img2_tensor):
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    # d = torch.rand(5)
    d = loss_fn_alex(img1_tensor, img2_tensor)

    return d


def short_term(pred_folder,gt_folder,model_type):

    total_ssim = []
    total_psnr = []
    total_lpips = []
    for i in range(32):
        pred_img_seq = []
        gt_img_seq = []
        for j in range(1,5):
            pred_img_path = f"{pred_folder}/{i}/{j}.png"
            gt_img_path = f"{gt_folder}/{i}/{j}.png"
            pred_img = Image.open(pred_img_path)
            gt_img = Image.open(gt_img_path)

            ssim_score = calc_ssim(pred_img,gt_img)
            total_ssim.append(ssim_score)

            psnr_score = calc_psnr(pred_img,gt_img)
            total_psnr.append(psnr_score)

            pred_img_tensor = default_transform(pred_img)
            pred_img_tensor = pred_img_tensor*2 - 1
            pred_img_seq.append(pred_img_tensor)

            gt_img_tensor = default_transform(gt_img)
            gt_img_tensor = gt_img_tensor*2 - 1
            gt_img_seq.append(gt_img_tensor)
        
        pred_img_seq = torch.stack(pred_img_seq,dim=0)
        gt_img_seq = torch.stack(gt_img_seq,dim=0)
        lpips_score = calc_lpips(pred_img_seq,gt_img_seq)
        total_lpips.append(torch.sum(lpips_score).detach().cpu().numpy())
    
    print(f"model = {model_type}")
    print(f"ssim score = {sum(total_ssim)/len(total_ssim)}")
    print(f"psnr score = {sum(total_psnr)/len(total_psnr)}")
    print(f"lpips score = {sum(total_lpips)/len(total_lpips)}")
    print("")

    

if __name__ == '__main__':
    gt_folder_path = "saved_video/gt"
    pred_folder_path = "saved_video/re10K_epipolar_1"

    pred_folder_list = [
        "saved_video/mae025_epoch210_inter1_randstart",
        "saved_video/epipolar_epoch470_inter1_samestart"
    ]

    for pred_folder_path in pred_folder_list:

        short_term( pred_folder=pred_folder_path,
                    gt_folder=gt_folder_path,
                    model_type = pred_folder_path.split('/')[-1]
                )
    
