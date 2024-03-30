import os
import glob
import torch
import utils
import cv2
import argparse
import time
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

from midas.model_loader import default_models, load_model


def run(model_path, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
        square=False, grayscale=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    image = Image.open("test_folder/test_0.png")
    image = np.array(image)
    image = image / 255.0       # (64,64,3)

    image = transform({"image": image})["image"] # (3,256,256)

    sample = torch.from_numpy(image).to(device).unsqueeze(0) # (1,3,256,256)
    prediction, src_l2, srcl3,srcl4 = model.forward(sample)

    # print(f"predict shape = {prediction.shape}")

    # prediction = (
    #     torch.nn.functional.interpolate(
    #         prediction.unsqueeze(1),
    #         size=(256,256),
    #         mode="bicubic",
    #         align_corners=False,
    #     )
    #     .squeeze()
    #     .cpu()
    #     .numpy()
    # )

    # depth_min = prediction.min()
    # depth_max = prediction.max()
    # depth = prediction
    # prediction = 255 * (depth - depth_min) / (depth_max - depth_min)    

    # prediction = prediction.astype(np.uint8)    

    # predict_img = Image.fromarray(prediction)
    # predict_img.save("test_folder/depth.png")


model_type = "dpt_swin2_tiny_256"
model_weights = default_models[model_type]

with torch.no_grad():
    run(model_weights,model_type)



