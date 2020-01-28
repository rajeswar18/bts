from __future__ import absolute_import, division, print_function

import os
import sys
import time
import argparse
import numpy as np

# Computer Vision
import cv2
from matplotlib import colors
from scipy import ndimage
from skimage.transform import resize, rescale

# Visualization
import matplotlib.pyplot as plt

# from bts.bts_model import BtsModel
from bts.models.bts_nyu_v2_pytorch_densenet161.bts_nyu_v2_pytorch_densenet161 import BtsModel
from bts.test_imgs import get_all_imgs, get_img, PATH


plasma = plt.get_cmap('plasma')
greys = plt.get_cmap('Greys')


class args:
    bts_size = 512
    model_name = "bts_inet_pytorch_train"
    encoder = "densenet121_bts"
    max_depth = 50
    checkpoint_path = "../bts/models/bts_inet_pytorch_train/model2.pth"
    input_height = 128
    input_width = 128
    dataset = "inet"
    mode = "test"
    fix_first_conv_blocks = False
    fix_first_conv_block = False

# class args:
#     bts_size = 512
#     model_name = "bts_nyu_v2_pytorch_densenet161"
#     encoder = "densenet161_bts"
#     max_depth = 50
#     checkpoint_path = "../bts/models/bts_nyu_v2_pytorch_densenet161/model"
#     input_height = 128
#     input_width = 128
#     dataset = "inet"
#     mode = "test"
#     fix_first_conv_blocks = False
#     fix_first_conv_block = False


import torch
from torch.autograd import Variable

model = BtsModel(params=args)
model = torch.nn.DataParallel(model)

checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
model.eval()

all_imgs = get_all_imgs()

fig, axs = plt.subplots(len(all_imgs), 3)
for idx,img in enumerate(all_imgs):
    input_image = plt.imread(get_img(img)).astype(np.float32)
    # # Normalize image
    scaling_factor = 128/input_image.shape[0]
    input_image = rescale(input_image, scaling_factor, anti_aliasing=False, multichannel=True).astype(np.float32)
    input_image_cropped = input_image[:128, :128, :]
    plotimg = np.copy(input_image_cropped)

    input_image_cropped[:, :, 0] = (input_image_cropped[:, :, 0] - 123.68) * 0.017
    input_image_cropped[:, :, 1] = (input_image_cropped[:, :, 1] - 116.78) * 0.017
    input_image_cropped[:, :, 2] = (input_image_cropped[:, :, 2] - 103.94) * 0.017


    input_images = np.expand_dims(input_image_cropped, axis=0)
    input_images = np.transpose(input_images, (0, 3, 1, 2))
    #
    with torch.no_grad():
        image = torch.from_numpy(input_images)
        focal = torch.tensor([600])
        # Predict
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_cropped = model(image, focal)

    depth = np.zeros((128, 128), dtype=np.float32)
    depth[:,:] = depth_cropped[0].cpu().squeeze() / args.max_depth
    norm = colors.Normalize(vmin=0, vmax=1)
    coloredDepth = (greys(norm(depth))[:, :, :3] * 255).astype('uint8')

    axs[idx, 0].imshow(plotimg.astype(np.uint8))
    axs[idx, 0].set_title('Input RGB')
    axs[idx, 0].axis("off")

    axs[idx, 1].imshow((depth*255).astype(np.uint8))
    axs[idx, 1].set_title('Depth')
    axs[idx, 1].axis("off")

    axs[idx, 2].imshow(coloredDepth)
    axs[idx, 2].set_title('Depth, but colored')
    axs[idx, 2].axis("off")

plt.tight_layout()
plt.show()
