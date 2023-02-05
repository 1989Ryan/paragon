import hydra
import torch
from paragon.models.paragon import ParaGon
from paragon.utils.grounding_utils import contrastive_loss_fn, loss_fn, get_bboxes_from_mask_torch, loss_fn_tar, loss_fn_tar_cross_entropy
from paragon.tabletop_dataset import tabletop_gym_obj_dataset
# from torch.utils.tensorboard import SummaryWriter
import os
import wandb

@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):
    device = torch.device(cfg['model']['device']) if torch.cuda.is_available() else torch.device('cpu')
    # from torch.utils.tensorboard import SummaryWriter
    wandb.init(project="ParaGon_train_n{}".format(cfg['dataset']['data_num']))
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    
    print(os.path.join(data_dir, '{}-train'.format(task)))
    print("loading data")
    dataset = tabletop_gym_obj_dataset(
        root=cfg['dataset']['data_dir'], 
        num=cfg['dataset']['data_num'])
    print('loading finish')
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                    factor=cfg['train']['factor'], patience=5,
                                    threshold=cfg['train']['threshold'], verbose=cfg['train']['verbose'], 
                                    min_lr=cfg['train']['min_lr']) 
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0)
    if cfg['dataset']['data_num'] <= 200:
        lr_scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,400,600], gamma=0.4)
    elif cfg['dataset']['data_num'] <= 2000:
        lr_scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.4)
    else:
        lr_scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45], gamma=0.4)
    # writer1 = SummaryWriter('tensorboard/target')
    # writer2 = SummaryWriter('tensorboard/placement')
    epoch_num = cfg['train']['epoch_num']
    model.train()
    counter = 0
    ave_dis_loss = 0.0
    ave_tar_loss = 0.0
    total_loss = []
    total_num = 0
    step_counter = 0
    for epoch in range(epoch_num):
        for data in data_loader:
            for cc in range(2):
                model.zero_grad()
                masks = data['masks']
                img = data['image']
                if cc == 0:
                    ins = data['ins_1']
                    labels = torch.vstack(data['label_place']).to(
                        cfg['model']['device']).transpose(0, 1)
                else:
                    ins = data['ins_2']
                    labels = torch.vstack(data['label_place_2']).to(
                        cfg['model']['device']).transpose(0, 1)
                assert not labels.isnan().any()
                bboxes = get_bboxes_from_mask_torch(masks)
                pred_, tar_prob, bbox_coord, text_gene_loss = model(ins, bboxes, img)
                pred, weights = pred_

                t_bbox = data['target_bbox']
                t_coord = torch.tensor([t_bbox[0]/2 + t_bbox[2]/2, t_bbox[1]/2 + t_bbox[3]/2], device=device)
                
                dis_tar_loss = loss_fn_tar_cross_entropy(
                    bbox_coord[0], tar_prob[0], t_coord
                )
                
                dis_loss = loss_fn(
                    (pred[:, -1], weights[:, -1]), 
                    labels/cfg['model']['position_size'],
                )
                

                loss = dis_loss + dis_tar_loss + 10 * text_gene_loss#+ con_loss
                loss.backward()
                optimizer.step()
                
                ave_dis_loss += dis_loss.detach()
                ave_tar_loss += dis_tar_loss.detach()
                total_loss.append(loss.detach())
            counter += 1
            if counter % 100 == 99:
                wandb.log({
                    'ave_dis_loss': ave_dis_loss.cpu().numpy()/200, 
                    'ave_tar_loss': ave_tar_loss.cpu().numpy()/200},
                    step= epoch * cfg['dataset']['data_num'] * 3 + 
                        (counter-99) * cfg['train']['batch_size'])
                ave_dis_loss = 0
                ave_tar_loss = 0
            total_num += 1
            if total_num % 3000 == 2999:
                ave_total = sum(total_loss) / len(total_loss)
                lr_scheduler.step(ave_total)
                total_loss = []
                total_num = 0
        counter = 0
        lr_scheduler_2.step(epoch)
        dir = './trained_model/ParaGon_{}'.format(cfg['dataset']['data_num'])
        if cfg['dataset']['data_num'] <= 200:
            if epoch in [80, 160, 240, 270, 300]:
                if not os.path.isdir(dir): 
                    os.makedirs(dir)
                torch.save(model.state_dict(), dir + '/{}_epoch.pt'.format(epoch))
        elif cfg['dataset']['data_num'] < 2000:
            if epoch in [30, 60, 75, 90, 100, 110]:
                if not os.path.isdir(dir): 
                    os.makedirs(dir)
                torch.save(model.state_dict(), dir + '/{}_epoch.pt'.format(epoch))
        else:
            if epoch in [4, 8, 10, 12, 14, 15]:
                if not os.path.isdir(dir): 
                    os.makedirs(dir)
                torch.save(model.state_dict(), dir + '/{}_epoch.pt'.format(epoch))
        if not os.path.isdir(dir): 
            os.makedirs(dir)
        torch.save(model.state_dict(), dir + '/last.pt')
    # writer1.close()
    # writer2.close()


if __name__=='__main__':
    main()