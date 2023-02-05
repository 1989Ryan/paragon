import os
import numpy as np
import torch
from PIL import Image

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

        

class tabletop_gym_box_dataset(torch.utils.data.Dataset):
    '''object detector dataset of tabletop gym'''
    def __init__(self, root, transform, test=False):
        self.root = root
        self.class_num = 2
        self.transforms = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        if test:
            list_dir_1 = listdir_fullpath('./dataset/test_4_obj_nvisii') 
            list_dir_2 = listdir_fullpath('./dataset/test_10_obj_nvisii')
            list_dir_3 = listdir_fullpath('./dataset/test_11_obj_nvisii')
  
        else:
            list_dir_1 = listdir_fullpath('./dataset/train_4_obj_nvisii') 
            list_dir_2 = listdir_fullpath('./dataset/train_10_obj_nvisii')
            list_dir_3 = listdir_fullpath('./dataset/train_11_obj_nvisii')
  
        self.paths = list_dir_1 + list_dir_2 + list_dir_3
    
    def __getitem__(self, idx):
        '''
        in the dataset we need to predefine several components
        '''
        # load images and masks
        img_path = os.path.join(self.paths[idx], "rgb.png")
        mask_path = os.path.join(self.paths[idx], "mask.png")
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[2:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        len_objs = len(obj_ids)
        boxes = []
        masks_idx = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin >= xmax or ymin >= ymax:
                len_objs -= 1
                masks_idx.append(False)
                continue
            else:
                boxes.append([xmin, ymin, xmax, ymax])
                masks_idx.append(True)
        # num_objs = len_objs
        labels = torch.zeros((len_objs, ), dtype=torch.int64)
        counter = 0
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin >= xmax or ymin >= ymax:
                continue
            else:
                labels[counter] = 1 # obj_ids[i] - obj_ids[0] + 1
                counter += 1

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float64)
        # there is only one class
        # labels = torch.zeros((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks[masks_idx], dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # assert len(boxes) == len(labels)
        assert len(boxes) == len(masks)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.paths)