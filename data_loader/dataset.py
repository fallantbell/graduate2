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

class ACIDdataset(Dataset):
    def __init__(self,data_root,mode):
        assert mode == 'train' or mode == 'test' or mode == 'validation'

        self.inform_root = '{}/acid/{}'.format(data_root, mode)
        self.image_root = '{}/acid_dataset/{}'.format(data_root, mode)

        self.transform = default_transform

        if mode == 'train':
            self.ordered = False
        else:
            self.ordered = True

        self.video_dirs = []
        self.image_dirs = []
        self.inform_dirs = []
        self.total_img = 0

        H = 144
        W = 176

        self.H = H
        self.W = W

        self.seq_len = 2        #* 一次取多少個image 

        for video_dir in os.listdir(self.image_root):
            self.video_dirs.append(video_dir)
            frame_namelist = []
            for frame in os.listdir(os.path.join(self.image_root, video_dir)):
                self.total_img+=1
                frame_namelist.append(frame)

            self.image_dirs.append(frame_namelist)
        
        for video in self.video_dirs:
            inform_path = '{}/{}.txt'.format(self.inform_root,video)

            frame_num = -1

            frame_list = []

            with open(inform_path, 'r') as file:
                for line in file:
                    frame_num+=1
                    if frame_num==0:
                        continue
                    frame_informlist = line.split()
                    frame_list.append(frame_informlist)
            
            self.inform_dirs.append(frame_list)

        print(f"video num: {len(self.video_dirs)}")
        print(f"total image: {self.total_img}")
        print(f"load {mode} data finish")
        print(f"-------------------------------------------")

        self.video_idx=0
        self.frame_idx=0

    def __len__(self):
        return len(self.video_dirs) #* 每個video 都會有frame-1 個data pair

    def get_image(self):
        if self.ordered:
            if self.video_idx == len(self.image_dirs) - 1:
                self.video_idx = 0
            else:
                self.video_idx += 1
        else:
            self.video_idx = np.random.randint(len(self.image_dirs))

        self.frame_idx = np.random.randint(len(self.image_dirs[self.video_idx])-self.seq_len)

        video_name = self.video_dirs[self.video_idx]

        image_seq = []

        for i in range(self.seq_len):
            frame_name = self.image_dirs[self.video_idx][self.frame_idx+i]
            img_path = '{}/{}/{}'.format(self.image_root,video_name,frame_name)
            img = Image.open(img_path)
            img = img.resize((self.W,self.H))
            img = self.crop_image(img)
            image_seq.append(self.transform(img))
        
        image_seq = torch.stack(image_seq)

        return image_seq


    def get_information(self):

        fx,fy,cx,cy = np.array(self.inform_dirs[self.video_idx][self.frame_idx][1:5], dtype=float)


        intrinsics = np.array([ [fx,0,cx,0],
                                [0,fy,cy,0],
                                [0,0,1,0],
                                [0,0,0,1]])
        
        #* unnormalize
        intrinsics[0] = intrinsics[0]*self.W
        intrinsics[1] = intrinsics[1]*self.H

        #* 調整 crop 後的 cx cy
        intrinsics[0][2] = intrinsics[0][2] * 3 / 5
        intrinsics[1][2] = intrinsics[1][2] * 3 / 5

        w2c_seq = []

        for i in range(self.seq_len):
            w2c = np.array(self.inform_dirs[self.video_idx][self.frame_idx+i][7:], dtype=float).reshape(3,4)
            w2c_4x4 = np.eye(4)
            w2c_4x4[:3,:] = w2c
            w2c_seq.append(torch.tensor(w2c_4x4))

        w2c_seq = torch.stack(w2c_seq)

        return intrinsics, w2c_seq
    
    def crop_image(self,img):
        original_width, original_height = img.size

        # 计算新的宽度和高度，保留3/5的长度和宽度
        new_width = original_width * 3 // 5
        new_height = original_height * 3 // 5

        # 截取图像，保留中心3/5的部分
        left = (original_width - new_width) // 2
        top = (original_height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        # 使用PIL的crop方法来截取图像
        cropped_img = img.crop((left, top, right, bottom))

        return cropped_img

    def __getitem__(self,index):

        img = self.get_image()

        intrinsics,w2c = self.get_information()

        result = {
            'img':img,
            'intrinsics':intrinsics,
            'w2c': w2c
        }

        return result


if __name__ == '__main__':
    test = ACIDdataset("../../../../disk2/icchiu","train")
    data = test[0]
    print(data['img'].shape)
    print(data['intrinsics'])
    print(data['w2c'][0])
    print(test.__len__())

    for i in range(data['img'].shape[0]):
        image = data['img'][i].numpy()
        image *= 255
        image = image.astype(np.uint8)
        image = rearrange(image,"C H W -> H W C")
        image = Image.fromarray(image)
        image.save(f"../test_folder/{i}.png")
