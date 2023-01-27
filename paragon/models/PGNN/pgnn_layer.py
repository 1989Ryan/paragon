import torch
import torch.nn as nn
from paragon.models.PGNN.deepset import ObservationFusion, ObservationFusionWeightUpdate
from torch import Tensor, LongTensor
from typing import Callable, Tuple, List
from paragon.utils.grounding_utils import find_edge_index_by_head, find_edge_index_by_tail
from paragon.models.grounding.node import MPNN_node
from torch_scatter import scatter

class PGNN_Layer_Base(nn.Module):
    def __init__(
        self,
        aggr: str,
        in_channel_dim: int,
        out_channel_dim: int,
        particle_num: int = 50,
        resamp_alpha: float = 0.5,
        device: str = 'cuda:0',
    ) -> None:
        super().__init__()
        self.aggr = aggr
        self.in_channel_dim = in_channel_dim
        self.out_channel_dim = out_channel_dim
        self.particle_num = particle_num
        self.resamp_alpha = resamp_alpha
        self.device=device
        

    def resampling(self, particles: Tensor, prob: Tensor):
        """
        The implementation of soft-resampling. We implement soft-resampling in a batch-manner.
        :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                        each tensor has a shape: [self.particle_num * batch_size, h_dim]
        :param prob: weights for particles in the log space. Each tensor has a shape: [self.particle_num * batch_size, 1]
        :return: resampled particles and weights according to soft-resampling scheme.
        """
        resamp_alpha = self.resamp_alpha
        resamp_prob = resamp_alpha * torch.exp(prob) + (1 - resamp_alpha) * 1 / self.particle_num 
        resamp_prob = resamp_prob.view(-1, self.particle_num)


        indices = torch.multinomial(resamp_prob,
                                    num_samples=self.particle_num, replacement=True)
        
        batch_size = indices.size(0)

        # indices = indices.transpose(1, 0).contiguous().to(self.device)
        offset = torch.arange(batch_size, device=self.device).unsqueeze(1)
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()


        particles_new = particles[flatten_indices].clone()

        prob_new = torch.exp(prob.clone().view(-1, 1)[flatten_indices]) + 1e-8
        prob_new = prob_new / (resamp_alpha * prob_new + (1 - resamp_alpha) / self.particle_num)
        prob_new = torch.log(prob_new).view(-1, self.particle_num)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=-1, keepdim=True)
        prob_new = prob_new.view(-1, 1)
        
        return particles_new, prob_new

    def reparameterize(self, mu: Tensor, var: Tensor):
        """
        Reparameterization trick
        :param mu: mean
        :param var: variance
        :return: new samples from the Gaussian distribution
        """
        std = torch.nn.functional.softplus(var)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.shape, device=self.device).normal_()
        else:
            eps = torch.FloatTensor(std.shape, device=self.device).normal_()
        return mu + eps * std
    
    def _aggregate(self, inputs: Tensor, index: LongTensor) -> Tensor:
        '''
        Args:
            inputs: (Tensor) message from neighbors and from observations
            index: (torch.LongTensor) indicating the index for message fusion
        '''
        return scatter(inputs, index, dim=1, reduce=self.aggr)

    def _aggregate_belief(
        self, 
        m_w_j2i: Tensor, 
        m_w_o2i: Tensor,
        e_idx: Tensor,
        o_idx: Tensor,
        ) -> Tensor:
        '''
        Args:
            :m_w_j2i: (Tensor) belief of message from node j to i, [b, n_e, n, 1]
            :m_w_o2i: (Tensor) belief of message from observation, [b, k-1, n, 1]
        Returns:
            :h_w: (Tensor) belief of aggregated node features, [b, k, n, 1]
        '''
        h_w = scatter(m_w_j2i, e_idx, dim=1, reduce='sum')
        h_w.index_add_(dim=1, index=o_idx, source=m_w_o2i)
        h_w_n = h_w - torch.logsumexp(h_w, dim=2, keepdim=True)
        return h_w_n
    
    def reshape_node_features(self, edge_idx: Tensor, features: Tensor):
        f_s = features.size(-1)
        idx_0 = edge_idx.unsqueeze(-1).unsqueeze(2).repeat(1, 1, self.particle_num, f_s)
        return torch.gather(features, 1, idx_0)

    def _compute_message(
        self, 
        h: Tuple[Tensor, Tensor],
        o: Tuple[Tensor, Tensor],
        e_f: Tensor,
        e_idx: Tensor,
    ) -> Tuple[
        Tuple[Tensor, Tensor],
        Tuple[Tensor, Tensor]
    ]:
        
        return NotImplementedError

    def _aggregate_message(
        self, 
        e_idx: Tensor,
        o_idx: Tensor,
        message_j2i: Tuple[Tensor, Tensor], 
        message_o2i: Tuple[Tensor, Tensor], 
    ) -> Tuple[Tensor, Tensor]:  
        
        return NotImplementedError
    
    def forward(
        self, 
        x, 
    ):
        h, o, e_f, e_idx, o_idx = x

        m_i, m_oi = self._compute_message(h, o, e_f, e_idx, o_idx)
        h_new = self._aggregate_message(e_idx, o_idx, m_i, m_oi)
        
        return (h_new, o, e_f, e_idx, o_idx)

class PGNN_Layer_Baseline(PGNN_Layer_Base):
    def __init__(
        self, 
        aggr: str, 
        in_channel_dim: int, 
        out_channel_dim: int, 
        particle_num: int = 50, 
        resamp_alpha: float = 0.5, 
        device: str = 'cuda:0'
    ) -> None:
        
        super().__init__(
            aggr, 
            in_channel_dim, 
            out_channel_dim, 
            particle_num, 
            resamp_alpha, 
            device
        ) 
        self.fn_m = nn.Linear(2 * self.in_channel_dim, self.out_channel_dim)
        self.fn_m_w = nn.Linear(3 * self.in_channel_dim, 1)
        self.fn_o_w = nn.Linear(2 * self.in_channel_dim, 1)
        self.fn_o = nn.Linear(self.in_channel_dim, self.out_channel_dim)
        self.fn_o_ = nn.Linear(self.in_channel_dim, self.out_channel_dim)
        self.fn_u = nn.Linear(self.in_channel_dim, self.out_channel_dim)
        self.layer_norm_m = nn.LayerNorm(self.in_channel_dim)
        self.layer_norm_o = nn.LayerNorm(self.in_channel_dim)
        self.layer_norm_u = nn.LayerNorm(self.in_channel_dim)
    
    def _compute_message(
        self, 
        h: Tuple[Tensor, Tensor],
        o: Tuple[Tensor, Tensor],
        e_f: Tensor,
        e_idx: Tensor,
        o_idx: Tensor,
    ) -> Tuple[
            Tuple[Tensor, Tensor], 
            Tuple[Tensor, Tensor]]:
        '''
        Args:
            h: (Tuple) hidden features, (h, h_w)
                h: [b, k, n, d]
                h_w: [b, k, n, 1]
            o: (Tuple) observations (o, o_w)
                o: [b, k, n, d]
                h_w: [b, k, n, 1]
            e_f: (Tensor) edge features, [b, n_e, d]
            e_idx: (Tensor) edge index for connection, [b, 2, n_e]
        '''
        h_f, h_w = h
        o_f, o_w = o
        bs = h_f.size(0)
        
        m_o_ = self.fn_o_(o_f)

        m_o      = self.layer_norm_o(m_o_.view(-1, self.in_channel_dim))\
            .view(bs, -1, self.particle_num, self.in_channel_dim).contiguous()
        
        m_o2i     = torch.nn.functional.leaky_relu(m_o)

        h_f_0 = self.reshape_node_features(e_idx[:, 0, :], h_f)
        h_f_w = self.reshape_node_features(e_idx[:, 0, :], h_w)
        
        m_n_ = self.fn_m(torch.cat([h_f_0, e_f.unsqueeze(2).repeat(1, 1, self.particle_num, 1)], dim=-1))
        m_n = self.layer_norm_m(m_n_.view(-1, self.in_channel_dim))\
            .view(bs, -1, self.particle_num, self.in_channel_dim).contiguous()
        m_j2i = torch.nn.functional.leaky_relu(m_n + h_f_0)
        
        return (m_j2i, h_f_w), (m_o2i, o_w)

    def _aggregate_message(
        self, 
        e_idx: Tensor,
        o_idx: Tensor,
        message_j2i: Tensor, 
        message_o2i: Tensor, 
    ) -> Tuple[Tensor, Tensor]:

        m_j2i, m_w_j2i = message_j2i
        m_o2i, m_w_o2i = message_o2i

        bs = m_j2i.size(0)
        
        m_2i = self._aggregate(m_j2i.view(bs, -1, self.particle_num, self.in_channel_dim), e_idx[:, 1, :])
        m_2i.index_add_(1, o_idx, m_o2i.view(bs, -1, self.particle_num, self.in_channel_dim))

        h_w_new = self._aggregate_belief(m_w_j2i, \
            m_w_o2i, e_idx[:, 1, :], o_idx)
        h_new = torch.nn.functional.leaky_relu(self.layer_norm_u(self.fn_u(m_2i).view(-1, self.in_channel_dim))\
            .view(bs, -1, self.particle_num, self.in_channel_dim).contiguous())

        return h_new.view(bs, -1, self.particle_num, self.in_channel_dim), \
            h_w_new.view(bs, -1, self.particle_num, 1)
    
class PGNN_Layer(PGNN_Layer_Base):
    def __init__(
        self, 
        aggr: str, 
        in_channel_dim: int, 
        out_channel_dim: int, 
        particle_num: int = 50, 
        resamp_alpha: float = 0.5, 
        device: str = 'cuda:0'
    ) -> None:
        super().__init__(
            aggr, 
            in_channel_dim, 
            out_channel_dim, 
            particle_num, 
            resamp_alpha, 
            device
        ) 
        self.fn_m = nn.Linear(self.in_channel_dim * 2, self.out_channel_dim)
        self.fn_o = nn.Linear(self.in_channel_dim, 2 * self.out_channel_dim)
        self.fn_o_ = nn.Linear(self.in_channel_dim, 2 * self.out_channel_dim)
        self.fn_u = nn.Linear(self.in_channel_dim, self.out_channel_dim)
        # self.batch_norm_m = nn.BatchNorm1d(self.particle_num)
        # self.batch_norm_o = nn.BatchNorm1d(self.particle_num)
        self.layer_norm = nn.LayerNorm(self.in_channel_dim)
        self.layer_norm_m = nn.LayerNorm(self.in_channel_dim)
        self.layer_norm_u = nn.LayerNorm(self.in_channel_dim)
        self.obs_fn_1 = ObservationFusionWeightUpdate(
            self.in_channel_dim, self.out_channel_dim)
        self.obs_fn_2 = ObservationFusion(
            self.in_channel_dim, self.out_channel_dim)

    def _compute_message(
        self, 
        h: Tuple[Tensor, Tensor],
        o: Tuple[Tensor, Tensor],
        e_f: Tensor,
        e_idx: Tensor,
        o_idx: Tensor,
    ) -> Tuple[
            Tuple[Tensor, Tensor], 
            Tuple[Tensor, Tensor]]:
        '''
        Args:
            h: (Tuple) hidden features, (h, h_w)
                h: [b, k, n, d]
                h_w: [b, k, n, 1]
            o: (Tuple) observations (o, o_w)
                o: [b, k, n, d]
                h_w: [b, k, n, 1]
            e_f: (Tensor) edge features, [b, n_e, d]
            e_idx: (Tensor) edge index for connection, [b, 2, n_e]
        '''
        h_f, h_w = h
        o_f, o_w = o
        bs = h_f.size(0)
        
        h_o = h_f[:, o_idx].clone()
        h_w_o = h_w[:, o_idx].clone()
        h_o_in = (h_o, h_w_o)
        h_w_new = self.obs_fn_1((h_o_in, o))
        h_w[:, o_idx] = h_w_new.clone()
        h_f_0 = self.reshape_node_features(e_idx[:, 0, :], h_f)
        h_w_0 = self.reshape_node_features(e_idx[:, 0, :], h_w)
        
        m_n = self.fn_m(torch.cat([h_f_0, e_f.unsqueeze(2).repeat(1, 1, self.particle_num, 1)], dim=-1))
        m_n = self.layer_norm_m(m_n.view(-1, self.in_channel_dim))\
            .view(bs, -1, self.particle_num, self.in_channel_dim).contiguous()
        m_j2i = torch.nn.functional.leaky_relu(m_n + h_f_0)
        
        return (m_j2i, h_w_0), (o_f, o_w)
    
    def _aggregate_belief(
        self, 
        m_w_j2i: Tensor, 
        # m_w_o2i: Tensor,
        e_idx: Tensor,
        # o_idx: Tensor,
        ) -> Tensor:
        '''
        Args:
            :m_w_j2i: (Tensor) belief of message from node j to i, [b, n_e, n, 1]
            :m_w_o2i: (Tensor) belief of message from observation, [b, k-1, n, 1]
        Returns:
            :h_w: (Tensor) belief of aggregated node features, [b, k, n, 1]
        '''
        h_w = scatter(m_w_j2i, e_idx, dim=1, reduce='sum')
        h_w_n = h_w - torch.logsumexp(h_w, dim=2, keepdim=True)
        return h_w_n
    
    def _aggregate_message(
        self, 
        e_idx: Tensor,
        o_idx: Tensor,
        message_j2i: Tensor, 
        message_o2i: Tensor, 
    ) -> Tuple[Tensor, Tensor]:

        m_j2i, m_w_j2i = message_j2i
        # m_o2i, m_w_o2i = message_o2i

        bs = m_j2i.size(0)
        
        m_2i = self._aggregate(m_j2i.view(bs, -1, self.particle_num, self.in_channel_dim), e_idx[:, 1, :])
        m_2i_with_obs = m_2i[:, o_idx].clone()
        m_2i_w = self._aggregate_belief(
            m_w_j2i.view(bs, -1, self.particle_num, 1), 
            e_idx[:, 1, :],
        )
        m_2i_w_with_obs = m_2i_w[:, o_idx].clone()
        m2i = (m_2i_with_obs, m_2i_w_with_obs)
        h_new_with_obs, h_w_new_with_obs = self.obs_fn_2((m2i, message_o2i))
        m_2i[:, o_idx] = h_new_with_obs.clone()        
        m_2i_w[:, o_idx] = h_w_new_with_obs.clone()
        
        h_new = torch.nn.functional.leaky_relu(self.layer_norm_u(self.fn_u(m_2i)\
            .view(-1, self.in_channel_dim))\
            .view(bs, -1, self.particle_num, self.in_channel_dim).contiguous())

        h_new_, h_w_new_ = self.resampling(h_new.view(-1, self.in_channel_dim), m_2i_w.view(-1, 1))

        return h_new_.view(bs, -1, self.particle_num, self.in_channel_dim), \
            h_w_new_.view(bs, -1, self.particle_num, 1)

class PGNN_Layer_No_RS(PGNN_Layer_Base):
    def __init__(
        self, 
        aggr: str, 
        in_channel_dim: int, 
        out_channel_dim: int, 
        particle_num: int = 50, 
        resamp_alpha: float = 0.5, 
        device: str = 'cuda:0'
    ) -> None:
        super().__init__(
            aggr, 
            in_channel_dim, 
            out_channel_dim, 
            particle_num, 
            resamp_alpha, 
            device
        ) 
        self.fn_m = nn.Linear(2 * self.in_channel_dim, 2 * self.out_channel_dim)
        self.fn_m_w = nn.Linear(3 * self.in_channel_dim, 1)
        self.fn_o_w = nn.Linear(2 * self.in_channel_dim, 1)
        self.fn_o = nn.Linear(self.in_channel_dim, 2 * self.out_channel_dim)
        self.fn_o_ = nn.Linear(self.in_channel_dim, 2 * self.out_channel_dim)
        self.fn_u = nn.Linear(self.in_channel_dim, self.out_channel_dim)
        self.batch_norm_m = nn.BatchNorm1d(self.particle_num)
        self.batch_norm_o = nn.BatchNorm1d(self.particle_num)
    
    def _compute_message(
        self, 
        h: Tuple[Tensor, Tensor],
        o: Tuple[Tensor, Tensor],
        e_f: Tensor,
        e_idx: Tensor,
        o_idx: Tensor,
    ) -> Tuple[
            Tuple[Tensor, Tensor], 
            Tuple[Tensor, Tensor]]:
        '''
        Args:
            h: (Tuple) hidden features, (h, h_w)
                h: [b, k, n, d]
                h_w: [b, k, n, 1]
            o: (Tuple) observations (o, o_w)
                o: [b, k, n, d]
                h_w: [b, k, n, 1]
            e_f: (Tensor) edge features, [b, n_e, d]
            e_idx: (Tensor) edge index for connection, [b, 2, n_e]
        '''
        h_f, h_w = h
        o_f, o_w = o
        bs = h_f.size(0)
        
        m_o_mu_var     = self.fn_o(o_f)
        m_o_mu_var_out = self.fn_o_(o_f)

        m_o_mu_out, m_o_var_out = torch.split(m_o_mu_var_out, split_size_or_sections=self.in_channel_dim, dim=-1)
        m_o_mu, m_o_var         = torch.split(m_o_mu_var, split_size_or_sections=self.in_channel_dim, dim=-1)
        
        m_o_out = self.reparameterize(m_o_mu_out, m_o_var_out).contiguous()
        m_o_    = self.reparameterize(m_o_mu, m_o_var).contiguous()

        # m_o_out_ = self.batch_norm_o(m_o_out.view(-1, self.particle_num, self.in_channel_dim))\
        #     .view(bs, -1, self.particle_num, self.in_channel_dim).contiguous()
        # m_o      = self.batch_norm_o(m_o_.view(-1, self.particle_num, self.in_channel_dim))\
        #     .view(bs, -1, self.particle_num, self.in_channel_dim).contiguous()
        
        m_o2i_out = torch.nn.functional.leaky_relu(m_o_out)
        m_o2i     = torch.nn.functional.leaky_relu(m_o_)

        h_f_ = h_f.index_add_(dim=1, index=o_idx, source=m_o2i)
        h_f_0 = self.reshape_node_features(e_idx[:, 0, :], h_f_)
        h_w_0 = self.reshape_node_features(e_idx[:, 0, :], h_w)
        o_w_0 = self.reshape_node_features(e_idx[:, 0, :][e_idx[:, 0, :]<o_f.size(1)].unsqueeze(0), o_w)
        
        m_n_mu_var = self.fn_m(torch.cat([h_f_0, e_f.unsqueeze(2).repeat(1, 1, self.particle_num, 1)], dim=-1))
        m_n_mu, m_n_var = torch.split(m_n_mu_var, split_size_or_sections=self.in_channel_dim, dim=-1)
        m_n_ = self.reparameterize(m_n_mu, m_n_var).contiguous()
        # m_n = self.batch_norm_m(m_n_.view(-1, self.particle_num, self.in_channel_dim))\
        #     .view(bs, -1, self.particle_num, self.in_channel_dim).contiguous()
        m_j2i = torch.nn.functional.leaky_relu(m_n_)

        m_w = h_w_0.exp().clone()
        m_w[:, e_idx[:, 0, :][e_idx[:, 0, :]<o_f.size(1)]] *= o_w_0
        h_w_new = m_w - torch.logsumexp(m_w, dim=2, keepdim=True)

        return (m_j2i, h_w_new), (m_o2i_out, o_w)

    def _aggregate_message(
        self, 
        e_idx: Tensor,
        o_idx: Tensor,
        message_j2i: Tensor, 
        message_o2i: Tensor, 
    ) -> Tuple[Tensor, Tensor]:

        m_j2i, m_w_j2i = message_j2i
        m_o2i, m_w_o2i = message_o2i

        bs = m_j2i.size(0)

        # m_j2i_new, m_w_j2i_new = self.resampling(m_j2i.view(-1, self.in_channel_dim), m_w_j2i.view(-1, 1))
        
        m_2i = self._aggregate(m_j2i.view(bs, -1, self.particle_num, self.in_channel_dim), e_idx[:, 1, :])
        m_2i.index_add_(1, o_idx, m_o2i.view(bs, -1, self.particle_num, self.in_channel_dim))
        
        h_w_new = self._aggregate_belief(m_w_j2i.view(bs, -1, self.particle_num, 1), \
            m_w_o2i.view(bs, -1, self.particle_num, 1), e_idx[:, 1, :], o_idx)
        h_new = torch.nn.functional.leaky_relu(self.fn_u(m_2i))

        return h_new.view(bs, -1, self.particle_num, self.in_channel_dim), \
            h_w_new.view(bs, -1, self.particle_num, 1)