import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from einops import rearrange

default_transform = transforms.Compose([
    transforms.ToTensor(),
])

class Re10k_dataset(Dataset):
    def __init__(self,data_root,mode,max_interval=5,midas_transform = None):
        assert mode == 'train' or mode == 'test' or mode == 'validation'

        self.inform_root = '{}/RealEstate10K/{}'.format(data_root, mode)
        self.image_root = '{}/realestate/{}'.format(data_root, mode)

        self.transform = default_transform
        self.midas_transform = midas_transform

        self.max_interval = max_interval   #* 兩張圖片最大的間隔


        self.video_dirs = []
        self.image_dirs = []
        self.inform_dirs = []
        self.total_img = 0

        #* 原始圖像大小
        H = 360
        W = 640

        #* 256 x 256 縮放版本
        H = 256
        W = 455

        #* 64 x 64 縮放版本
        H = 64
        W = 114

        self.H = H
        self.W = W

        self.square_crop = True     #* 是否有做 center crop 

        xscale = W / min(H, W)      #* crop 之後 cx cy 的縮放比例
        yscale = H / min(H, W)

        dim = min(H, W)

        self.xscale = xscale
        self.yscale = yscale


        for video_dir in os.listdir(self.image_root):
            self.video_dirs.append(video_dir)

        print(f"video num: {len(self.video_dirs)}")
        print(f"load {mode} data finish")
        print(f"-------------------------------------------")


    def __len__(self):
        return len(self.video_dirs) 
    
    def get_image(self,index):

        #* 選哪一個video
        video_idx = index

        #* 讀取video 每個frame 的檔名
        frame_namelist = []
        video_dir = self.video_dirs[video_idx]
        npz_file_path = f"{self.image_root}/{video_dir}/data.npz"
        npz_file = np.load(npz_file_path)

        for file_name in sorted(npz_file.files):
            frame_namelist.append(file_name)

        if len(frame_namelist) <= self.max_interval:
            return None, None, False


        #* 隨機取間距
        self.interval_len = np.random.randint(self.max_interval) + 1
        # print(f"interval len = {self.interval_len}")

        #* 隨機取origin frame
        self.frame_idx = np.random.randint(len(frame_namelist)-self.interval_len)
        # print(f"origin idx = {self.frame_idx}, target idx = {self.frame_idx+self.interval_len}")

        image_seq = []
        frame_idx = [self.frame_idx, self.frame_idx+self.interval_len]  #* 兩張圖片, 一個origin 一個target

        cnt = 0
        for idx in frame_idx:
            frame_name = frame_namelist[idx]
            img_np = npz_file[frame_name]
            # print(f"img ori shape:{img_np.shape}")
            img = Image.fromarray(img_np)
            img = img.resize((self.W,self.H))
            img = self.crop_image(img)
            if cnt == 0:
                src_img_numpy = np.array(img)
                src_img_numpy = src_img_numpy / 255.0 
                src_img = self.midas_transform({"image": src_img_numpy})["image"]
                src_img_tensor = torch.from_numpy(src_img)

            img_tensor = self.transform(img)
            img_tensor = img_tensor*2 - 1
            image_seq.append(img_tensor)
            cnt += 1
        
        image_seq = torch.stack(image_seq)

        return image_seq, src_img_tensor, True


    def get_information(self,index):

        #* 讀取選定video 的 information txt
        video_idx = index
        video_dir = self.video_dirs[video_idx]
        inform_path = '{}/{}.txt'.format(self.inform_root,video_dir)

        frame_num = -1
        frame_list = []

        with open(inform_path, 'r') as file:
            for line in file:
                frame_num+=1
                if frame_num==0:
                    continue
                frame_informlist = line.split()
                frame_list.append(frame_informlist)

        fx,fy,cx,cy = np.array(frame_list[self.frame_idx][1:5], dtype=float)


        intrinsics = np.array([ [fx,0,cx,0],
                                [0,fy,cy,0],
                                [0,0,1,0],
                                [0,0,0,1]])
        
        #* unnormalize
        # intrinsics[0] = intrinsics[0]*self.W
        # intrinsics[1] = intrinsics[1]*self.H

        #* 調整 crop 後的 cx cy
        # if self.square_crop:
        #     intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
        #     intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

        c2w_seq = []
        frame_idx = [self.frame_idx, self.frame_idx+self.interval_len]
        # print(f"inform idx = {frame_idx}")

        for idx in frame_idx:
            c2w = np.array(frame_list[idx][7:], dtype=float).reshape(3,4)
            c2w_4x4 = np.eye(4)
            c2w_4x4[:3,:] = c2w
            c2w_seq.append(torch.tensor(c2w_4x4))

        c2w_seq = torch.stack(c2w_seq)

        intrinsics = torch.from_numpy(intrinsics)

        return intrinsics, c2w_seq
    
    def crop_image(self,img):
        original_width, original_height = img.size

        # center crop 的大小
        new_width = min(original_height,original_width)
        new_height = new_width

        # 保留中心的crop 的部分
        left = (original_width - new_width) // 2
        top = (original_height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        # 使用PIL的crop方法来截取图像
        cropped_img = img.crop((left, top, right, bottom))

        return cropped_img

    def __getitem__(self,index):

        img, src_img_tensor, good_video = self.get_image(index)

        #* video frame 數量 < max interval 
        if good_video == False:
            print(f"false")
            return self.__getitem__(index+1)

        intrinsics,c2w = self.get_information(index)

        result = {
            'img':img,
            'src_img': src_img_tensor,
            'intrinsics':intrinsics,
            'c2w': c2w
        }

        return result


if __name__ == '__main__':
    test = Re10k_dataset("../../../disk2/icchiu","train")
    data = test[0]
    print(data['img'].shape)
    print(data['intrinsics'])
    print(data['w2c'][0])
    print(test.__len__())
    print(test.interval_len)

    for i in range(data['img'].shape[0]):
        image = data['img'][i].numpy()
        image = (image+1)/2
        image *= 255
        image = image.astype(np.uint8)
        image = rearrange(image,"C H W -> H W C")
        image = Image.fromarray(image)
        image.save(f"../test_folder/test_{i}.png")
