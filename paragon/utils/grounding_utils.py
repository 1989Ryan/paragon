import torch
import torch.nn as nn
from torchvision import transforms
import json
import numpy as np
from torch import Tensor, LongTensor
from PIL import Image

def find_edge_index_by_head(edge_index, index):
    '''
    given a node index i, find the indexes of all edges like [i, j]
    args:
        edge_index: torch.tensor with shape of [2, N]
    '''
    head_edge_index = edge_index[0, :]
    return torch.isin(head_edge_index, index)

def find_edge_index_by_tail(edge_index, index):
    '''
    given a node index i, find the indexes of all edges like [j, i]
    args:
        edge_index: torch.tensor with shape of [2, N]
    '''
    tail_edge_index = edge_index[1, :]
    return torch.isin(tail_edge_index, index)

class Gaussian_model(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.mu = nn.parameter.Parameter(
            data=torch.zeros((dim), dtype=torch.float32), 
            requires_grad=True)
        self.sigma = nn.parameter.Parameter(
            data=torch.diag(torch.ones((dim), dtype=torch.float32)), 
            requires_grad=True)

    def forward(self, x):
        return self.mu + x @ self.sigma

def conv(batchNorm, in_channels, out_channels, kernel_size=3, stride=1,
        dropout=0.0):
    if batchNorm:
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, padding=(kernel_size-1)//2, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(dropout)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.ReLU(inplace=True),
                # nn.LayerNorm(out_channels),
                nn.Dropout2d(dropout)
                )

clip_preprocess = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

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

def get_bboxes_coord(bboxes):
    output_list = []
    for bbox in bboxes:
        x_coord = bbox[0] / 2 + bbox[2] / 2
        y_coord = bbox[1] / 2 + bbox[3] / 2
        output_list.append([x_coord, y_coord])
    return torch.tensor(output_list)

def from_bboxes_to_image_cliport(bboxes, image):
    # print(image.size())
    img_boxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        # print(xmin)
        img_box = image[:,
            max(0, int(ymin)): min(319, int(ymax)), 
            max(0, int(xmin)): min(159, int(xmax)),
            ]
        img_boxes.append(img_box)
    return img_boxes

def from_bboxes_to_image(bboxes, image):
    # print(image.size())
    img_boxes = []
    for bbox in bboxes:
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        xmin, ymin, xmax, ymax = bbox
        # print(xmin)
        img_box = image[:, 
            max(0, int(ymin) - 12): min(639, int(ymax) + 12), 
            max(0, int(xmin) - 12): min(639, int(xmax) + 12)]
        img_boxes.append(img_box)
    return img_boxes

def get_triplet(dep, dep_label):
    triplet = []
    for label in dep_label:
        # print(label[0])
        subj = dep[int(label[0])][1]
        rel = dep[int(label[1])][1]
        obj = dep[int(label[2])][1]
        triplet.append([subj, rel, obj])
    return triplet

def bbox_to_tensor(size, bbox, norm_fn, resize_fn):
    if isinstance(size, tuple):
        mapping = torch.zeros([size[0], size[1]])
    else: 
        mapping = torch.zeros([size, size])
    xmin, ymin, xmax, ymax = bbox
    mapping[int(ymin):int(ymax), int(xmin):int(xmax)] = 1.0
    mapping = mapping.unsqueeze(dim=0).unsqueeze(dim=0)
    mapping = resize_fn.forward(mapping)
    mapping = norm_fn.forward(mapping).float()
    return mapping

def bboxes_to_coord(bboxes):
    stacking = []
    for bbox in bboxes:
        stacking.append(torch.tensor([bbox[0]/2 + bbox[2]/2, bbox[1]/2 + bbox[3]/2]))
    coord_tensors = torch.vstack(stacking)
    return coord_tensors

def bboxes_to_pos_matrix(size, bboxes, n_fn, r_fn):
    stacking = []
    for bbox in bboxes:
        pos_tensor = bbox_to_tensor(size, bbox, n_fn, r_fn)
        stacking.append(pos_tensor)
    bbox_tensors = torch.vstack(stacking)
    return bbox_tensors

def loss_fn_tar_cross_entropy(coords: torch.Tensor, prob: torch.Tensor, target: torch.Tensor):
    diff = (coords - target).square().sum(dim=-1).sqrt()
    _, idx = torch.min(diff, dim=-1, keepdim=True)
    a = torch.zeros_like(prob)
    a[:, idx] = 1.0
    return torch.nn.functional.cross_entropy(prob, a)


def loss_fn_tar(coord, target):
    # assert coord.size()==target.size(), print(coord.size(), target.size())
    loss_l2 =  torch.nn.functional.mse_loss(
        coord.reshape(1, 2), target.float(), reduction='sum')
    # loss_l1 = torch.nn.functional.l1_loss(
    #     coord, target.float().unsqueeze(0), reduction='none')
    loss = loss_l2  #+ 2e-1 * loss_l1

    return loss

def loss_fn(pred, target):
    coord, prob = pred
    particle_num = coord.size(1)

    prob = torch.exp(prob)
    # exp_coord = (coord * prob).sum(dim=-2)

    loss_l2 = torch.nn.functional.mse_loss(coord, target.float().view(-1, 1, 2).repeat(1, particle_num, 1), reduction='none')
    # print(loss_l2)
    exp_loss_l2 = (-loss_l2 * 141.4).sum(dim=-1, keepdim=True).exp()
    # print(exp_loss_l2)
    # print(prob.size())
    weighted_sum_exp_loss_l2 = (exp_loss_l2 * prob).sum(dim=-2) + 1e-8
    # print(weighted_sum_exp_loss_l2)
    log_sum_loss = -weighted_sum_exp_loss_l2.log()

    # loss_l2 = torch.sum(prob * torch.nn.functional.mse_loss(
    # loss_l1 = torch.nn.functional.l1_loss(exp_coord, target.float(), reduction='sum')
    #     coord, target.float().unsqueeze(1).repeat(1, particle_num, 1), reduction='none').sum(dim=-1), dim=-1)
    # loss_l1 = torch.sum(prob * torch.nn.functional.l1_loss(
    #     coord, target.float().unsqueeze(1).repeat(1, particle_num, 1), reduction='none').sum(dim=-1), dim=-1)
    # loss = loss_l2 
    # + 2e-1 * loss_l1
    return log_sum_loss

def contrastive_loss_fn(pred, target):
    coord, prob = pred
    particle_num = coord.size(1)
    
    prob = torch.exp(prob) + 1e-8
    loss_l2 = torch.nn.functional.mse_loss(
        coord, target.float().unsqueeze(1).repeat(1, particle_num, 1), reduction='none').sum(dim=-1)
    de = torch.log(torch.sum(prob, dim=1))
    v, _ = torch.min(loss_l2, dim=-1, keepdim=True)
    nu = torch.log(torch.sum(prob[loss_l2<1.2*v], dim=1))
    loss = de - nu
    return loss.mean()

def get_bboxes_from_mask_torch(mask):
    batch_size = mask.size(0)
    obj_ids = torch.unique(mask)
    obj_ids = obj_ids[2:]
    masks = mask == obj_ids[:, None, None, None]
    # masks = masks.transpose(0, 1)
    num_objs = len(obj_ids)
    bboxes = [] 
    b_bboxes = []
    for b in range(batch_size):
        for i in range(num_objs):
            pos = torch.nonzero(masks[i, b])
            if pos.size(0) == 0: continue
            xmin = torch.min(pos[:, 1])
            xmax = torch.max(pos[:, 1])
            ymin = torch.min(pos[:, 0])
            ymax = torch.max(pos[:, 0])
            if (xmin >= xmax).any() or (ymin >= ymax).any():
                continue
            else:
                bboxes.append([xmin, ymin, xmax, ymax])
        b_bboxes.append(bboxes)
        bboxes = []
    return b_bboxes

def get_bboxes_from_mask(mask, obj_keys):
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[2:]
    masks = mask == obj_ids[:, None, None]
    num_objs = len(obj_ids)
    bbox_dict = {}
    # num_objs = len_objs
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        if xmin >= xmax or ymin >= ymax:
            continue
        else:
            bbox_dict[obj_keys[int(obj_ids[i] - obj_ids[0] + 1)]] = [int(xmin), int(ymin), int(xmax), int(ymax)]
    return bbox_dict

def resampling(particles: Tensor, prob: Tensor, particle_num: int, resamp_alpha: float, device: str):
    """
    The implementation of soft-resampling. We implement soft-resampling in a batch-manner.
    :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                    each tensor has a shape: [self.particle_num * batch_size, h_dim]
    :param prob: weights for particles in the log space. Each tensor has a shape: [self.particle_num * batch_size, 1]
    :return: resampled particles and weights according to soft-resampling scheme.
    """
    assert not prob.isnan().any()
    resamp_prob = resamp_alpha * torch.exp(prob) + (1 - resamp_alpha) * 1 / prob.size(1)
    # resamp_prob = resamp_prob.view(prob.size(0), -1)
    indices = torch.multinomial(resamp_prob,
                                num_samples=particle_num, replacement=True)

    flatten_indices = indices.view(-1, 1).squeeze() # % particles.size(0)

    particles_new = particles[flatten_indices]

    prob_new = torch.exp(prob.reshape(-1, 1)[flatten_indices])
    prob_new = prob_new / (resamp_alpha * prob_new + (1 - resamp_alpha) / particle_num)
    prob_new = torch.log(prob_new).view(-1, particle_num)
    prob_new = prob_new - torch.logsumexp(prob_new, dim=-1, keepdim=True)
    prob_new = prob_new.view(-1, 1)
    if torch.cuda.is_available():
        particles_new = particles_new.to(device)
        prob_new = prob_new.to(device)
    
    return particles_new, prob_new

def reparameterize(mu: Tensor, var: Tensor, device: str):
    """
    Reparameterization trick
    :param mu: mean
    :param var: variance
    :return: new samples from the Gaussian distribution
    """
    std = torch.nn.functional.softplus(var)
    if torch.cuda.is_available():
        eps = torch.cuda.FloatTensor(std.shape, device=device).normal_()
    else:
        eps = torch.FloatTensor(std.shape, device=device).normal_()
    return mu + eps * std