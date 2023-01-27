import torch
import torch.nn as nn
from torch.nn import LayerNorm
import numpy as np

class Visual_Object_Grounding(nn.Module):
    """visual grounding module for object bbox grounding"""
    """require CLIP module"""
    def __init__(self, device_name, v_embd_dim, w_embd_dim):
        super().__init__()
        device = device_name if torch.cuda.is_available() else "cpu"
        self.device = device
        scale = v_embd_dim ** -0.5
        w_scale = w_embd_dim ** -0.5
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))    
        self.proj_v = nn.Parameter(torch.empty(v_embd_dim, w_embd_dim))
        self.v_ln_post = LayerNorm(v_embd_dim)
        self.t_ln_post = LayerNorm(w_embd_dim)
        self.text_projection = nn.Parameter(torch.empty(w_embd_dim, w_embd_dim))
        # self.linear = torch.nn.Linear(v_embd_dim, w_embd_dim, bias=False, device=self.device)
        # self.linear.apply(self._init_weights)
        torch.nn.init.normal_(self.proj_v, std=w_embd_dim ** -0.5)
        torch.nn.init.normal_(self.text_projection, std=w_embd_dim ** -0.5)
    
    def encode_v_f(self, v):
        v_n = self.v_ln_post(v)
        return v_n @ self.proj_v
    
    def encode_t_f(self, t):
        t_n = self.t_ln_post(t)
        return t_n @ self.text_projection

    def mul(self, v, t):
        v_ = v / v.norm(dim=-1, keepdim=True)
        t_ = t / t.norm(dim=-1, keepdim=True)
        return (self.logit_scale.exp() * v_ @ t_.T).transpose(0, 1)
    
    def forward(self, text_feature, visual_features):
        '''input: object referenced expression, and a set of bboxes in the img'''
        '''output: pro(x_visual|w_object), a vector of prob'''
        visual_feature = list(map(self.encode_v_f, visual_features)) # [bs x n, dim]
        text_f = list(map(self.encode_t_f, text_feature))
        assert not text_f[0].isnan().any(), print(self.text_projection)
        assert not visual_feature[0].isnan().any(), print(self.proj_v)
        semantic_logprob = list(map(self.mul, visual_feature, text_f)) # [bs x m, dim] @ [bs x 4, dim].T = [bs x m, bs x n]
        return semantic_logprob, visual_features

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.2)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class Visual_Object_Grounding_Bayesian(nn.Module):
    """visual grounding module for object bbox grounding"""
    """require CLIP module"""
    def __init__(self, device_name, embd_dim):
        super().__init__()
        device = device_name if torch.cuda.is_available() else "cpu"
        self.device = device
        self.embd_dim = embd_dim
        self.linear = torch.nn.Linear(self.embd_dim, self.embd_dim, bias=False, device=self.device)
        # self.linear.apply(self._init_weights)

    def mul(self, v, t):
        v_ = v / v.norm(dim=-1, keepdim=True)
        t_ = t / t.norm(dim=-1, keepdim=True)
        return (1e2 * v_ @ t_.T).transpose(0, 1)
    
    def forward(self, text_feature, visual_features, prob):
        '''input: object referenced expression, and a set of bboxes in the img'''
        '''output: pro(x_visual|w_object), a vector of prob'''
        visual_feature = list(map(self.linear, visual_features)) # [bs x n, dim]
        prob_ = list(map(self.mul, visual_feature, text_feature)) # [bs x m, dim] @ [bs x 4, dim].T = [bs x m, bs x n]
        # print(semantic_logprob)
        prob_n = prob_ - torch.logsumexp(prob_, dim=-1, keepdim=True)
        semantic_logprob = prob_n + prob
        return semantic_logprob, visual_features

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.2)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)