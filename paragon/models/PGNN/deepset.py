import torch
import torch.nn as nn
from typing import Tuple
from paragon.utils.grounding_utils import conv

class ObservationFusionWeightUpdate(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.build_model()
    
    def build_model(self):
        # self.nn = nn.Sequential(
        #     nn.Linear(5* self.in_dim, self.in_dim),
        #     # nn.LayerNorm(self.in_dim)
        # )
        self.nn_2 = nn.Sequential(
            nn.Linear(self.in_dim * 2, 1)
        )
        self.nn_3 = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            # nn.LayerNorm(self.in_dim)
        )
    
    def forward(self, inputs: Tuple):
        features, obs = inputs
        
        f, w = features
        f_o, w_o = obs # [bs, nl, particle_num, embd_dim]
        bs, nl, pn, d = f.size()
        
        # f_o_tile = f_o.view(bs, nl, 1, pn, d).repeat(1, 1, pn, 1, 1)
        # f_tile = f.view(bs, nl, pn, 1, d).repeat(1, 1, 1, pn, 1)
        # w_o_tile = w_o.view(bs, nl, 1, pn, 1).repeat(1, 1, pn, 1, 1)
        
        # f_cat = torch.cat([f_o_tile, f_tile], dim=-1)
        # w_update = self.nn2(f_update)
        f_update = self.nn_3((f_o * w_o.exp()).sum(dim=2, keepdim=True))
        # w_update = (self.nn(f_cat) * w_o_tile.exp()).sum(dim=3)
        w_update = self.nn_2(torch.cat([f, f_update.repeat(1, 1, pn, 1)], dim=-1))
        w_new = w + w_update
        w_new_ = w_new - w_new.logsumexp(dim=-2, keepdim=True)
        # print(w_new_.size())
        return w_new_

class ObservationFusion(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relu = nn.LeakyReLU()
        self.build_model()
    
    def build_model(self):
        # self.nn = nn.Sequential(
        #     nn.Linear(6 * self.in_dim, self.in_dim),
        #     # nn.LayerNorm(self.in_dim)
        # )
        self.nn2 = nn.Sequential(
            nn.Linear(self.in_dim * 2, 1)
        )
        self.nn3 = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            # nn.LayerNorm(self.out_dim)
        )
    
    def forward(self, inputs: Tuple):
        features, obs = inputs
        
        f, w = features
        f_o, w_o = obs # [bs, nl, particle_num, embd_dim]
        bs, nl, pn, d = f.size()
        
        # f_o_tile = f_o.view(bs, nl, 1, pn, d).repeat(1, 1, pn, 1, 1)
        # f_tile = f.view(bs, nl, pn, 1, d).repeat(1, 1, 1, pn, 1)
        # w_o_tile = w_o.view(bs, nl, 1, pn, 1).repeat(1, 1, pn, 1, 1)
        
        # f_cat = torch.cat([f_o_tile, f_tile], dim=-1)
        f_update = self.nn3((f_o * w_o.exp()).sum(dim=2, keepdim=True))
        w_update = self.nn2(torch.cat([f, f_update.repeat(1, 1, pn, 1)], dim=-1))
        f_new = f + f_update
        w_new = w + w_update
        w_new_ = w_new - w_new.logsumexp(dim=-2, keepdim=True)
        # print(w_new_.size())
        return f_new, w_new_
        
class ExpectedSum(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relu = nn.LeakyReLU()
        self.build_model()
    
    def build_model(self):
        self.nn = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.LayerNorm(self.out_dim)
        )

    def forward(self, inputs: Tuple):
        x, w = inputs
        E_x = (x * w.exp()).sum(dim=-2, keepdim=True)
        x_out = self.relu(self.nn(E_x))
        return x_out, w
