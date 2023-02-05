from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
from skimage import io
import os
import numpy as np
import torch
from PIL import Image, ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

class CLIPort_Dataset(Dataset):
    def __init__(self, root, mode):
        # checkpointdir = os.checkpointdir.join(root, mode)
        assert mode in ['train', 'val', 'test']
        self.root = root
        self.mode = mode
        assert os.path.exists(root), 'Path {} does not exist'.format(root)

        if mode == 'val':
            list_dir = listdir_fullpath(self.root) 
            self.img_paths = random.sample(list_dir, 200)
        
        else:
            list_dir = listdir_fullpath(root)
            self.img_paths = list(sorted(list_dir))
        
    def __getitem__(self, index):
        img = io.imread(self.img_paths[index])[:, :, :3]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        img = transform(img)
        
        return img
    
    def __len__(self):
        return len(self.img_paths)