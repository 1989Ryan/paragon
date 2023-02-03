
from paragon.models.PGNN.pgnn_layer import PGNN_Layer
from paragon.utils.grounding_utils import conv, Gaussian_model
from typing import List, Tuple
from torch import Tensor
import torch
import torch.nn as nn

from paragon.utils.grounding_utils import resampling

class PGNN_Base(nn.Module):
    def __init__(
        self,
        aggr: str, 
        num_layers: int,
        embd_dim: int,
        particle_num: int,
        output_dim: int,
        resamp_alpha: float = 0.5,
        position_size: int = 640, 
        word_embd_dim: int = 512,
        device: str = 'cuda:0',
    ) -> None:
        super().__init__()
        self.device = device
        self.particle_num = particle_num
        self.position_size = position_size
        self.embd_dim = embd_dim
        self.output_dim = output_dim
        self.aggr = aggr
        self.word_embd_dim = word_embd_dim
        self.resamp_alpha = resamp_alpha
        self.num_layers = num_layers
        self.__build_model()
    
    def __build_model(self):
        self.init_obs = nn.Linear(self.embd_dim, 2 * self.embd_dim)
        self.gaussian_init = Gaussian_model(self.embd_dim) 
        self.placement_decoder = nn.Linear(self.embd_dim, self.output_dim)
        self.relation_encoder = nn.Sequential(
            nn.Linear(self.word_embd_dim, self.embd_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.embd_dim * 2, self.embd_dim),
            )
        self.reverse_relation_encoder = nn.Sequential(
            nn.Linear(self.word_embd_dim, self.embd_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.embd_dim * 2, self.embd_dim),
        )
    
    def __init_hidden_features(self, bs, nl):
        '''
        Returns:
            :h: (Tuple), (h_v, h_w)
                h_v (Tensor) [bs, k, p, d]
                h_w (Tensor) [bs, k, p, 1]
        '''
        h_v = torch.normal(
                mean=0.0, std=0.2, 
                size=(bs, nl, self.particle_num, self.embd_dim),
                device=self.device
            )
        # h_v = self.gaussian_init(h_n)
        h_w = torch.log(torch.ones(size = (bs, nl, self.particle_num, 1), device=self.device) / self.particle_num)

        return (h_v, h_w)
    

    # def __build_pmfgnn_model(self):
    #     self.model = None
    #     return NotImplementedError

    # def __build_obs_model(self):
    #     self.obs_encoder = None
    #     return NotImplementedError

    def forward(self, obs: List, rel_lang_embd: Tensor):
        '''
        Args:
            :obs: (List[Tuple]) [bs, (Tensor, weights)]
            :rel_lang_embd: (Tensor) relation embeddings
                [bs, 4, word_embd_dim]
        Returns:
            :tar_pos: (Tensor) target pose [bs, output_dim]
        '''
        obs_v, obs_w = obs

        bs = len(obs_v)

        new_obs, new_obs_w = map(list,zip(*[\
            resampling(
                v, w, 
                particle_num=self.particle_num, 
                resamp_alpha=self.resamp_alpha, 
                device=self.device
            )\
            for v, w in zip(obs_v, obs_w)]))

        os = (160, 160) 
        t_obs = torch.stack(new_obs)\
            .view(-1, 1, os[0], os[1]).to(self.device) # [bs, (k-1)xp, m, m]
        
        
        t_w_obs = torch.stack(new_obs_w)\
            .view(bs, -1, self.particle_num, 1).to(self.device) # [bs, k-1, 50, 1]

        nl = int(t_obs.size(1) / self.particle_num)
        
        rel_embd = self.relation_encoder(rel_lang_embd).repeat_interleave(2, dim=1) # [bs, n_e/2, d]
        # print(t_obs.size())
        h_obs = self.obs_encoder(t_obs)\
            .view(bs, -1, self.particle_num, self.embd_dim) # [bs, k-1, p, d]
        
        o_in = (h_obs, t_w_obs)
        h_init = self.__init_hidden_features(bs, nl)

        e_idx = torch.tensor(
            [list(range(nl - 1)) + [nl - 1 for _ in range(nl - 1)], 
            [nl - 1 for _ in range(nl - 1)] + list(range(nl - 1))],
            device=self.device
        )

        o_idx = torch.tensor(
            list(range(nl - 1))
        ).to(self.device)

        h_out, _, _, _, _ = self.model(
            h_init, o_in, rel_embd, e_idx, o_idx)

        p_out = self.placement_decoder(h_out[:, -1])

        return p_out

class PGNN(PGNN_Base):
    def __init__(
        self, 
        aggr: str, 
        num_layers: int, 
        embd_dim: int, 
        particle_num: int, 
        output_dim: int, 
        resamp_alpha: float = 0.5, 
        position_size: int = 640, 
        word_embd_dim: int = 512, 
        device: str = 'cuda:0'
        ) -> None:
        super().__init__(aggr, num_layers, embd_dim, particle_num, output_dim, resamp_alpha, position_size, word_embd_dim, device)
        self.__build_obs_model()
        self.__build_pmfgnn_model()
    
    def __build_pmfgnn_model(self):
        # _model = []
        # for _ in range(self.num_layers):
        #     _model.append(New_PMFGNN_MPNN_Fusion(
        #         self.aggr, 
        #         self.embd_dim, 
        #         self.embd_dim, 
        #         self.particle_num, 
        #         self.resamp_alpha, 
        #         self.device))
        self.model = PGNN_Layer(
                self.aggr, 
                self.embd_dim, 
                self.embd_dim, 
                self.particle_num, 
                self.resamp_alpha, 
                self.device).to(self.device)

    def __build_obs_model(self):
        self.obs_encoder = nn.Sequential(
            conv(False, 1, 16, kernel_size=3, stride=2, dropout=0.2),
            conv(False, 16, 16, kernel_size=4, stride=3, dropout=0.2),
            conv(False, 16, 16, kernel_size=4, stride=3, dropout=0.2),
            conv(False, 16, 16, kernel_size=4, stride=3),
            nn.Flatten(),
            nn.Linear(144, self.embd_dim),
            nn.ReLU()
        ).to(self.device)


    def __init_hidden_features(self, bs, nl):
        '''
        Returns:
            :h: (Tuple), (h_v, h_w)
                h_v (Tensor) [bs, k, p, d]
                h_w (Tensor) [bs, k, p, 1]
        '''
        h_v = torch.normal(
                mean=0.0, std=0.2, 
                size=(bs, nl, self.particle_num, self.embd_dim),
                device=self.device
            )
        # h_v = self.gaussian_init(h_n)
        h_w = torch.log(torch.ones(size = (bs, nl, self.particle_num, 1), device=self.device) / self.particle_num)

        return (h_v, h_w)
    
    def forward(self, obs: List, rel_lang_embd: Tensor):
        '''
        Args:
            :obs: (List[Tuple]) [bs, (Tensor, weights)]
            :rel_lang_embd: (Tensor) relation embeddings
                [bs, 4, word_embd_dim]
        Returns:
            :tar_pos: (Tensor) target pose [bs, output_dim]
        '''
        obs_v, obs_w = obs

        bs = len(obs_v)
        sl = len(obs_v[0])

        os = (160, 160) 
        t_obs = torch.stack(obs_v)\
            .view(-1, 1, os[0], os[1]).to(self.device) # [bs, (k-1)xp, m, m]

        t_w_obs = torch.stack(obs_w)\
            .view(bs, -1, sl, 1).to(self.device) # [bs, k-1, 50, 1]

        # new_obs, new_obs_w = map(list,zip(*[\
        #     resampling(
        #         v, w, 
        #         particle_num=self.particle_num, 
        #         resamp_alpha=self.resamp_alpha, 
        #         device=self.device
        #     )\
        #     for v, w in zip(obs_v, obs_w)]))

        # os = (160, 160) 
        # t_obs = torch.stack(new_obs)\
            # .view(-1, 1, os[0], os[1]).to(self.device) # [bs, (k-1)xp, m, m]

        # t_w_obs = torch.stack(new_obs_w)\
            # .view(bs, -1, self.particle_num, 1).to(self.device) # [bs, k-1, 50, 1]

        nl = t_w_obs.size(1) + 1

        rel_embd = self.relation_encoder(rel_lang_embd) # [bs, n_e/2, d]
        rev_rel_embd = self.reverse_relation_encoder(rel_lang_embd)
        rel = torch.cat([rel_embd, rev_rel_embd], dim=1)
        # print(t_obs.size())
        h_obs = self.obs_encoder(t_obs)\
            .view(bs, -1, sl, self.embd_dim) # [bs, k-1, p, d]
        
        o_in = (h_obs, t_w_obs)
        h_init = self.__init_hidden_features(bs, nl)

        e_idx =torch.tensor([
            [list(range(nl - 1)) + [nl - 1 for _ in range(nl - 1)], 
            [nl - 1 for _ in range(nl - 1)] + list(range(nl - 1))]
            for _ in range(bs)],
            device=self.device
        )

        o_idx = torch.tensor(
            list(range(nl - 1))
        ).to(self.device)
        x = (h_init, o_in, rel, e_idx, o_idx)
            
        output_history = []
        h_v_out, h_w_out = h_init
        p_f = self.placement_decoder(h_v_out)
        p_out = torch.tanh(p_f)/2 + 0.5
        output_history.append((p_out, h_w_out))
        for _ in range(self.num_layers):
            x = self.model(x)

            h_out, _, _, _, _ = x
            
            h_v_out, h_w_out = h_out

            p_f = self.placement_decoder(h_v_out)
            
            p_out = torch.tanh(p_f)/2 + 0.5
            output_history.append((p_out, h_w_out))
        return p_out, h_w_out, output_history
