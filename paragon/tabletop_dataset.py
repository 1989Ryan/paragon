import torch
import os
import numpy as np
from PIL import Image
import json
from torchvision.transforms import functional as F
from torchvision.io import read_image
import random
import cv2
# from tabletop_gym.envs.data.template import instruction_template
# import math

def read_json(filepath):
    '''
    from filepath to instruction list
    :return:instruction list
    '''
    try:
        with open(filepath) as f:
            data = json.load(f)
    except IOError as exc:
        raise IOError("%s: %s" % (filepath, exc.strerror))
    return data

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

class tabletop_gym_obj_dataset(torch.utils.data.Dataset):
    '''object detector dataset of tabletop gym'''
    def __init__(self, root, num=None, test=False):
        self.root = root
        self.class_num = 19
        # load all image files, sorting them to
        # ensure that they are aligned
        if not test:
            list_dir_1 = listdir_fullpath(root + '/train_4_obj_nvisii') 
            list_dir_2 = listdir_fullpath(root + '/train_10_obj_nvisii')
            list_dir_3 = listdir_fullpath(root + '/train_11_obj_nvisii')
        else:
            list_dir = listdir_fullpath(root) 
        self.test = test
        if num is not None:
            self.paths = random.sample(list_dir_1, int(num)) \
                + random.sample(list_dir_2, int(num)) \
                + random.sample(list_dir_3, int(num))
        else:
            self.paths = list(sorted(list_dir))
        self.info_simple = [read_json(os.path.join(ele, "info_simple.json")) for ele in self.paths]
        self.info_compositional = [read_json(os.path.join(ele, "info_compositional.json")) for ele in self.paths]

    
    def __getitem__(self, idx):
        '''
        in the dataset we need to predefine several components
        '''
        # load images and masks
        img_path = os.path.join(self.paths[idx], "rgb.png")
        mask_path = os.path.join(self.paths[idx], "mask.png")
        # info_path = os.path.join(self.paths[idx], "info_simple.json")
        # info_comp_path = os.path.join(self.paths[idx], "info_compositional.json")

        info = self.info_simple[idx]
        info_comp = self.info_compositional[idx]
        ins = info['instruction']
        ins_comp = info_comp['instruction']
        # simple_ins = info['simple_instruction']
        # complex_ins = info['complex_instruction']
        # img = read_image(img_path)
        # print(img)
        mask = read_image(mask_path)

        img = Image.open(img_path).convert("RGB")
        img = F.pil_to_tensor(img)
        sample = {}
       
        sample["masks"] = mask
        sample['label_place']= info['goal_pixel']
        sample['label_place_2'] = info_comp['goal_pixel']
        sample['bboxes'] = info['bboxes']
        sample['target_bbox'] = info['target_bbox']
        if self.test:
            sample['relations_1'] = info["relations"]
            sample['relations_2'] = info_comp["relations"]
        sample["ins_1"] = ins
        sample["ins_2"] = ins_comp
        sample["image"] = F.convert_image_dtype(img)
        sample['raw_img'] = img_path
        return sample

    def __len__(self):
        return len(self.paths)