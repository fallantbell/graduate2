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

        self.inform_root = '{}/ACID/{}'.format(data_root, mode)
        self.image_root = '{}/ACID_IMG/{}'.format(data_root, mode)

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

        # 按照 inf 的大小
        H = 160
        W = 256

        self.H = H
        self.W = W

        #* 一次取所有的image 太久了，改成讀取path，get item 時再慢慢讀取
        for video_dir in os.listdir(self.image_root):
            self.video_dirs.append(video_dir)

            # frame_namelist = []
            # for frame in os.listdir(os.path.join(self.image_root, video_dir)):
            #     self.total_img+=1
            #     frame_namelist.append(frame)

            # self.image_dirs.append(frame_namelist)
        
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
        print(f"load {mode} data finish")
        print(f"-------------------------------------------")

        self.video_idx=0

    def __len__(self):
        return len(self.video_dirs) #* 多少個影片

    def get_image(self,idx):
        # if self.ordered:
        #     if self.video_idx == len(self.image_dirs) - 1:
        #         self.video_idx = 0
        #     else:
        #         self.video_idx += 1
        # else:
        #     self.video_idx = np.random.randint(len(self.image_dirs))

        # self.frame_idx = np.random.randint(len(self.image_dirs[self.video_idx])-self.seq_len)

        # video_name = self.video_dirs[self.video_idx]

        video_dir = self.video_dirs[idx]
        npz_path = '{}/{}/data.npz'.format(self.image_root,video_dir)
        data = np.load(npz_path)
        key = list(data.keys())
        video_len = len(key)

        max_distance = 5

        #* 兩張圖片間隔多長 [1~5]
        img_distance = np.random.randint(max_distance)+1
        start_idx = np.random.randint(video_len-img_distance)
        end_idx = start_idx + img_distance

        image_seq = []

        #* 影片太短
        if video_len < max_distance+1:
            return image_seq, False, start_idx,end_idx

        start_img = data[key[start_idx]]
        start_img = Image.fromarray(start_img)
        start_img = start_img.resize((self.W,self.H))
        image_seq.append(self.transform(start_img))

        end_img = data[key[end_idx]]
        end_img = Image.fromarray(end_img)
        end_img = end_img.resize((self.W,self.H))
        image_seq.append(self.transform(end_img))

        # for i in range(self.seq_len):
        #     frame_name = self.image_dirs[self.video_idx][self.frame_idx+i]
        #     img_path = '{}/{}/{}'.format(self.image_root,video_name,frame_name)
        #     img = Image.open(img_path)
        #     img = self.crop_image(img)
        #     img = img.resize((self.W,self.H))
        #     image_seq.append(self.transform(img))
        
        image_seq = torch.stack(image_seq)

        return image_seq, True, start_idx,end_idx


    def get_information(self,idx,start_idx,end_idx):

        fx,fy,cx,cy = np.array(self.inform_dirs[idx][start_idx][1:5], dtype=float)


        intrinsics = np.array([ [fx,0,cx,0],
                                [0,fy,cy,0],
                                [0,0,1,0],
                                [0,0,0,1]])
        
        #* unnormalize
        intrinsics[0] = intrinsics[0]*self.W
        intrinsics[1] = intrinsics[1]*self.H

        w2c_seq = []

        # for i in range(self.seq_len):
        #     w2c = np.array(self.inform_dirs[self.video_idx][self.frame_idx+i][7:], dtype=float).reshape(3,4)
        #     w2c_4x4 = np.eye(4)
        #     w2c_4x4[:3,:] = w2c
        #     c2w_4x4 = np.linalg.inv(w2c_4x4)
        #     w2c_seq.append(torch.tensor(w2c_4x4))

        #* start image extrinsic
        w2c = np.array(self.inform_dirs[idx][start_idx][7:], dtype=float).reshape(3,4)
        w2c_4x4 = np.eye(4)
        w2c_4x4[:3,:] = w2c
        c2w_4x4 = np.linalg.inv(w2c_4x4)
        w2c_seq.append(torch.tensor(w2c_4x4))

        #* end image extrinsic
        w2c = np.array(self.inform_dirs[idx][end_idx][7:], dtype=float).reshape(3,4)
        w2c_4x4 = np.eye(4)
        w2c_4x4[:3,:] = w2c
        c2w_4x4 = np.linalg.inv(w2c_4x4)
        w2c_seq.append(torch.tensor(w2c_4x4))

        w2c_seq = torch.stack(w2c_seq)

        return intrinsics, w2c_seq
    
    def crop_image(self,img):
        # 用來切除上下黑邊

        width, height = img.size

        top = 22
        bottom = 121

        # 使用PIL的crop方法来截取图像
        cropped_img = img.crop((0, top, width, bottom+1))

        return cropped_img

    def __getitem__(self,index):

        img, is_ok, start_idx,end_idx = self.get_image(index)

        #* 影片不夠長, 重選
        if is_ok == False:
            return self.__getitem__(np.random.randint(index))

        intrinsics,w2c = self.get_information(index,start_idx,end_idx)

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
