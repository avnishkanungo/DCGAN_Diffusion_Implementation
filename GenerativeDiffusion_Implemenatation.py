#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys
import torch
import torchvision.transforms as tvt
from PIL import Image
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvt
from torchvision import utils
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import random
import time
import skimage.io as io
from pycocotools.coco import COCO
import copy
from scipy.ndimage import zoom
import torch.optim as optim
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import time
import logging

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


# In[ ]:


# This code has been picked up from the DL Studio Library
# Diffusion Image Generation using the provided model weights
import os
import sys
import numpy as np

import torch

sys.path.append("/content/drive/MyDrive/DLStudio-2.4.2/DLStudio")
sys.path.append("/content/drive/MyDrive/DLStudio-2.4.2/GenerativeDiffusion")

from GenerativeDiffusion import *

gauss_diffusion   =  GaussianDiffusion(
                        num_diffusion_timesteps = 1000,

                    )


network =  UNetModel(
                       in_channels=3,
                       model_channels   =  128,
                       out_channels     =  3,
                       num_res_blocks   =  2,
                       attention_resolutions =  (4, 8),               ## for 64x64 images
                       channel_mult     =    (1, 2, 3, 4),            ## for 64x64 images
                       num_heads        =    1,
                       attention        =    True            ## <<< Must be the same as for RunCodeForDiffusion.py
#                       attention        =    False          ## <<< Must be the same as for RunCodeForDiffusion.py

                     )


top_level = GenerativeDiffusion(
                        gen_new_images        =        True,
                        image_size            =        64,
                        num_channels          =        128,
                        ema_rate              =        0.9999,
                        diffusion = gauss_diffusion,
                        network = network,
                        ngpu = 1,
                        path_saved_model = "/content/drive/MyDrive/DLStudio-2.4.2/ExamplesDiffusion/saved_model/", ##using the provided diffusion model weights
                        clip_denoised=True,
                        num_samples=1000,
                        batch_size_image_generation=4,
             )

if sys.argv[1] == '--model_path':
    model_path = sys.argv[2]


network.load_state_dict( torch.load("/content/drive/MyDrive/DLStudio-2.4.2/ExamplesDiffusion/saved_model/diffusion.pt") )

network.to(top_level.device)
network.eval()

print("sampling...")
all_images = []

while len(all_images) * top_level.batch_size_image_generation < top_level.num_samples:
    sample = gauss_diffusion.p_sampler_for_image_generation(
        network,
        (top_level.batch_size_image_generation, 3, top_level.image_size, top_level.image_size),
        device = top_level.device,
        clip_denoised = top_level.clip_denoised,
    )
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    gathered_samples = [sample]
    all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
    print(f"created {len(all_images) * top_level.batch_size_image_generation} samples")

arr = np.concatenate(all_images, axis=0)
arr = arr[: top_level.num_samples]

shape_str = "x".join([str(x) for x in arr.shape])
out_path = os.path.join("/content/drive/MyDrive/DLStudio-2.4.2/ExamplesDiffusion", f"samples_{shape_str}.npz")

np.savez(out_path, arr)

print("image generation completed")


# In[ ]:


# This code has been picked up from the DL Studio Library

data = np.load("/content/drive/MyDrive/DLStudio-2.4.2/ExamplesDiffusion/samples_2048x64x64x3.npz")

print("\n\n[visualize_sample.py]  the data object: ", data)                                       ## NpzFile 'RESULTS/samples_8x64x64x3.npz' with keys: arr_0
print("\n\n[visualize_sample.py]  type of the data object: ", type(data))                         ## <class 'numpy.lib.npyio.NpzFile'>
print("\n\n[visualize_sample.py]  shape of the object data['arr_0']: ", data['arr_0'].shape)      ## (8, 64, 64, 3)

for i, img in enumerate(data['arr_0']):
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(f"/content/drive/MyDrive/DLStudio-2.4.2/ExamplesDiffusion/visualize_samples/test_{i}.jpeg")


# In[ ]:


def list_files_in_folder(folder_path):
    files = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            files.append(os.path.join(folder_path, file_name))
    return files

real_folder_path = '/content/drive/MyDrive/DLStudio-2.4.2/ExamplesAdversarialLearning/celeba_dataset_64x64/0'
fake_folder_path_1 = '/content/drive/MyDrive/DLStudio-2.4.2/ExamplesAdversarialLearning/results_DG1/gen_fake_images1'

real_image_paths = list_files_in_folder(real_folder_path)
fake_image_paths_diffusion = list_files_in_folder(fake_folder_diffusion)


# In[ ]:


from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)
m1 , s1 = calculate_activation_statistics(real_image_paths , model , device = device )
m2 , s2 = calculate_activation_statistics(fake_image_paths_diffusion , model , device = device )
fid_value = calculate_frechet_distance(m1 , s1 , m2 , s2)
print(f'FID: { fid_value: .2f}')


# In[ ]:


# Displaying sample of generated images
def display_random_images(folder_path, num_images=16, rows=4, cols=4):
    files = os.listdir(folder_path)
    random_files = random.sample(files, num_images)
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        # Open and display each image
        img_path = os.path.join(folder_path, random_files[i])
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')  
    plt.show()

folder_path_diffusion = "/content/drive/MyDrive/DLStudio-2.4.2/ExamplesDiffusion/visualize_samples"
display_random_images(folder_path_diffusion)

