from object_detector.space.model import get_model
from object_detector.space.eval import get_evaluator
from object_detector.space.dataset import get_dataset, get_dataloader
from object_detector.space.model.space.space import Space
from object_detector.space.solver import get_optimizers
from object_detector.space.utils import Checkpointer, MetricLogger
from object_detector.space.engine.utils import get_config
from object_detector import Tabletop_Dataset
from object_detector.space.eval.ap import convert_to_boxes
from skimage import io
import os
import os.path as osp
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from object_detector.space.vis import get_vislogger
import time
from torch.nn.utils import clip_grad_norm_
import tqdm
import torch
from torchvision import transforms
from object_detector.run import objDetector
from object_detector.cliport_dataset_space import CLIPort_Dataset
import matplotlib.pyplot as plt
from matplotlib import patches
dataset = CLIPort_Dataset('/home/zirui/paraground/dataset/put-block-in-bowl-seen-colors-train/rgb/', 'val')
import numpy as np

data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0)

model = objDetector('space', 'cuda:0')
device='cuda:0'

with torch.no_grad():
    print('Computing boxes...')
    img =  io.imread('dataset/put-block-in-bowl-seen-colors-train/rgb/000001-2.pkl_0.png')[:, :, :3]
    fig, ax = plt.subplots()
    ax.imshow(img)

    # transform = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize((128, 128)),
    #         transforms.ToTensor(),
    #     ])
    # imgs = transform(img)
    # imgs = imgs.to(device)
    bboxes = model.run(img) 
    bbox_array = bboxes[0]
    print(bbox_array)
    for bbox in bbox_array:
        rect = patches.Rectangle((bbox[0], bbox[3]), bbox[2]-bbox[0], bbox[1]-bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.savefig('demo.png')