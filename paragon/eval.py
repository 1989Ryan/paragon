import torch
from paragon.tabletop_dataset import tabletop_gym_obj_dataset
from paragon.utils.grounding_utils import contrastive_loss_fn, get_bboxes_from_mask_torch, loss_fn
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
from paragon.models.paragon import ParaGon
from typing import Dict
import json
from object_detector.run import maskrcnn_obj_detecotr
from tqdm import tqdm
from time import sleep
import hydra


SPATIAL_RELATIONS = {
    0:[0, 0], # center
    1:[-1, 0], # left
    2:[1, 0], # right
    3:[0, -1], # front
    4:[0, 1], # behind
    5:[-1, -1], # front left
    6:[-1, 1], # upper left
    7:[1, -1], # front right
    8:[1, 1], # upper right 
}

def check_spatial_relations(bboxes: Dict, relation, pred: torch.Tensor, cfg):
    '''
    args:
        bboxes: bounding boxes with object name
        relation: spatial relations
        pred: the predicted placement
    '''
    flag_all = True
    for ctx, rel in relation:
        obj_bbox = bboxes[str(ctx.item())]
        ctx_coord = torch.tensor([obj_bbox[0]/2 + obj_bbox[2]/2, obj_bbox[1]/2 + obj_bbox[3]/2], device=cfg['model']['device'], dtype=torch.float)
        direction = pred - ctx_coord
        if isinstance(rel, list):
            flag = torch.tensor(
                [(direction * torch.tensor(SPATIAL_RELATIONS[rel_.item()], device=cfg['model']['device'])>=0).all() 
                for rel_ in rel]).all()
            if not flag:
                flag_all = False
        else:
            # print(direction)
            # print(SPATIAL_RELATIONS[rel.item()])
            if rel == 0:
                dis = direction.square().sum().sqrt()
                if dis > 40:
                    flag_all = False
            else:
                flag = (direction * torch.tensor(SPATIAL_RELATIONS[rel.item()], device=cfg['model']['device']) >=0).all()
                # flag = torch.matmul(direction, torch.tensor(SPATIAL_RELATIONS[rel.item()], device=args.dev_name, dtype=torch.float)) >= 0
                if not flag:
                    flag_all = False
    return flag_all

def dump_json(dict_data, filepath):
    with open(filepath, 'w') as fp:
        json.dump(dict_data, fp, indent=2)

def succ(coord, log_prob, groundtruth, bboxes, relation, args):
    if log_prob is None:
        mean_coord = coord
        return check_spatial_relations(bboxes, relation, mean_coord * 640, args)
    else:
        succ = 0
        prob = torch.exp(log_prob).detach()
        particle_num = coord.size()[1]
        for i in range(particle_num):
            if check_spatial_relations(bboxes, relation, (coord[0, i] * 640).detach(), args):
                succ += prob[0, i].item()
        return succ

@hydra.main(config_path="./cfg", config_name='eval')
def main(cfg):
    device = torch.device(cfg['model']['device']) if torch.cuda.is_available() else torch.device('cpu')
    model = ParaGon(
            aggr=cfg['model']['aggr'],
            word_embd_dim=cfg['model']['word_embd_dim'],
            embd_dim=cfg['model']['embd_dim'],
            gnn_layer_num=cfg['model']['layer_num'],
            particle_num=cfg['model']['particle_num'],
            resamp_alpha=cfg['model']['resamp_alpha'],
            position_size=cfg['model']['position_size'],
            device=cfg['model']['device'],
            return_loss=True,
    )
    model.to(device)
    model.eval()
    f1 = torch.load(cfg['model']['file'], map_location=device)
    model.load_state_dict(f1)
    total_num = 0
    succ_num = 0
    repeat = 1
    obj_detect = maskrcnn_obj_detecotr('/home/zirui/paraground/trained_model/0_9_mask_rcnn.pt')
    
    if cfg['dataset']['comp']:
        print('testing tasks: 10 objects with compositional instructions')
    else:
        print('testing tasks: 10 objects with simple instructions')
    dataset = tabletop_gym_obj_dataset('/home/zirui/paraground/dataset/test_10_obj_nvisii', test=True)
    data_n = len(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0)
    total_num = 0
    succ_num = 0
    pbar = tqdm(data_loader)
    for i, data in enumerate(pbar):
        for _ in range(repeat):
            model.zero_grad()
            masks = data['masks']
            img = data['image']
            bboxes, scores = obj_detect.query(img)
            
            if not cfg['dataset']['comp']:
                ins = data['ins_1']
                labels = torch.vstack(data['label_place']).to(device).transpose(0, 1)
                relations = data['relations_1']
            else:
                ins = data['ins_2']
                labels = torch.vstack(data['label_place_2']).to(device).transpose(0, 1)
                relations = data['relations_2']
            bboxes_dict = data['bboxes']
            assert not labels.isnan().any()
            pred_, tar_w, coord_tensor,_ = model(ins, [bboxes], img)
            pred, weights = pred_
            if not cfg['dataset']['comp']:
                succ_num += succ(pred[:, -1], weights[:, -1], data["label_place"], bboxes_dict, relations, cfg)
            else:
                succ_num += succ(pred[:, -1], weights[:, -1], data["label_place_2"], bboxes_dict, relations, cfg)
            
            total_num += 1
        pbar.set_postfix({"succ": succ_num/total_num})
    print("[{}/{}] succ: {}".format(i+1, data_n, succ_num/total_num))
    print("########## evaluation finish #############")

if __name__ == '__main__':
    main()