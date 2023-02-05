from .atari import Atari
from .obj3d import Obj3D
from torch.utils.data import DataLoader
from object_detector import CLIPort_Dataset


__all__ = ['get_dataset', 'get_dataloader']

def get_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test']
    return CLIPort_Dataset(cfg.dataset_roots.TABLE, mode)

def get_dataloader(cfg, mode):
    assert mode in ['train', 'val', 'test']
    
    batch_size = getattr(cfg, mode).batch_size
    shuffle = True if mode == 'train' else False
    num_workers = getattr(cfg, mode).num_workers
    
    dataset = get_dataset(cfg, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader
    
