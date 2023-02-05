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


class Tabletop_Dataset(Dataset):
    def __init__(self, root, mode):
        # checkpointdir = os.checkpointdir.join(root, mode)
        assert mode in ['train', 'val', 'test']
        self.root = root
        self.mode = mode
        assert os.path.exists(root), 'Path {} does not exist'.format(root)

        if mode == 'val':
            list_dir_1 = listdir_fullpath('/home/zirui/tabletop_gym/dataset/test_4_obj_simple') 
            list_dir_2 = listdir_fullpath('/home/zirui/tabletop_gym/dataset/test_10_obj_simple')
            list_dir_3 = listdir_fullpath('/home/zirui/tabletop_gym/dataset/test_11_obj_simple')
            list_dir = list_dir_1 + list_dir_2 + list_dir_3
            self.img_paths = random.sample(list_dir, 200)
        
        else:
            list_dir_1 = listdir_fullpath('/home/zirui/tabletop_gym/dataset/{}_4_obj_simple'.format(mode)) 
            list_dir_2 = listdir_fullpath('/home/zirui/tabletop_gym/dataset/{}_10_obj_simple'.format(mode))
            list_dir_3 = listdir_fullpath('/home/zirui/tabletop_gym/dataset/{}_11_obj_simple'.format(mode))
            list_dir = list_dir_1 + list_dir_2 + list_dir_3
            self.img_paths = list(sorted(list_dir))
        
    @property
    def bb_path(self):
        return self.img_paths
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_paths[index], "rgb.png")
        img = io.imread(img_path)[:, :, :3]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        img = transform(img)
        
        return img
    
    def __len__(self):
        return len(self.img_paths)