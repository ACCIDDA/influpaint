import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch

# import epiframework
import numpy as np
import pandas as pd
from influpaint.models import nn_blocks, ddpm, inpaint_module
from influpaint import ground_truth, plotting, converters, helpers, season_axis
from influpaint.utils import ground_truth, plotting, converters, helpers, season_axis
from torch.utils.data import DataLoader
from torchvision import transforms
import datetime

import sys

sys.path.append("CoPaint4influpaint")
from guided_diffusion import O_DDIMSampler
from guided_diffusion import unet
from utils import config

image_size = 64
channels = 1
batch_size = 512
epoch = 800
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
if device == "cuda":
    print(myutils.cuda_mem_info())
    torch.cuda.empty_cache()  # make sure we don't keep old stuff
    print(myutils.cuda_mem_info())

do_inpainting = True
do_training = False


from importlib import reload

ground_truth = reload(ground_truth)
if do_inpainting:
    if False:  # if True, take a training data sample as gt:
        inpaintfrom_idx = 19
        gt_keep_mask = np.ones((channels, image_size, image_size))
        gt_keep_mask[:, inpaintfrom_idx:, :] = 0
        # mask is ones for the known pixels, and zero for the ones to be infered
        gt = data.getitem_nocast(4)
        print(gt.shape)
        show_tensor_image(gt, place=ipl)
        plt.show()

    else:
        gt1 = ground_truth.GroundTruth(
            season_first_year="2024",
            data_date=datetime.datetime.today(),  # datetime.datetime(2024, 12, 3),
            mask_date=datetime.datetime.today(),
            channels=channels,
            image_size=image_size,
        )
        gt1.plot_mask()
        gt1.plot()
