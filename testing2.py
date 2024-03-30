import torch
from model.diffusion_model import Unet, GaussianDiffusion
from data_loader.re10k_dataset import Re10k_dataset
from einops import rearrange
from PIL import Image
import numpy as np

if __name__ == '__main__':

    test = Re10k_dataset("../../../disk2/icchiu","train")
    data = test[0]
    img = data['img']
    intrinsic = data['intrinsics']
    w2c = data['w2c']

    for i in range(2):
        image = img[i].numpy()
        image = (image+1)/2
        image *= 255
        image = image.astype(np.uint8)
        image = rearrange(image,"C H W -> H W C")
        image = Image.fromarray(image)
        image.save(f"test_folder/test_{i}.png")


    # unet_model = Unet(
    #     dim = 128,
    #     init_dim = 128,
    #     dim_mults = (2, 4, 8),
    #     channels=3, 
    #     out_dim=3,
    # )

    # device = "cuda"

    # unet_model = unet_model.to(device)

    # input = torch.randn(1, 3, 64, 64).to(device)

    # t = torch.randint(0, 1000, (1,), device=device).long()

    # output = unet_model(input,t,None)

    # print(f"output shape = {output.shape}")

