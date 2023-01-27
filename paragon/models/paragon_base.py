import torch
import torch.nn as nn
import clip
import timm

from typing import List, Tuple
from torch import Tensor
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Normalize, Resize

from transformers import DistilBertTokenizer, DistilBertModel

from paragon.models.SoftParsing.dep_tree import dep_tree_parser, extract_dep_edge_info
from paragon.models.VisualGrounding import Visual_Object_Grounding

BICUBIC = InterpolationMode.BICUBIC

class ParaGonBase(nn.Module):
    def __init__(
        self,
        aggr: str, 
        word_embd_dim: int,
        embd_dim: int,
        gnn_layer_num: str,
        particle_num: int,
        resamp_alpha: float, 
        position_size: int,
        device: str,
        ) -> None:
        super().__init__()
        self.device = device
        self.word_embd_dim = word_embd_dim
        self.particle_num = particle_num
        self.embd_dim = embd_dim
        self.dep_parser = dep_tree_parser()      
        self.gnn_layer_num = gnn_layer_num
        self.resamp_alpha = resamp_alpha
        self.position_size = position_size
        self.aggr = aggr
        self.r_fn = Resize((160, 160), interpolation=BICUBIC)
        self.n_fn = Normalize((0.5), (1.0))
    
    def forward(
        self, 
        text: List, 
        bboxes: List, 
        image: Tensor,
        ):
        return NotImplementedError

class ParaGonClipBase(ParaGonBase):
    def __init__(
        self, 
        aggr: str, 
        word_embd_dim: int, 
        embd_dim: int, 
        gnn_layer_num: str, 
        particle_num: int, 
        resamp_alpha: float, 
        position_size: int, 
        device: str
        ) -> None:
        super().__init__(
            aggr, 
            word_embd_dim, 
            embd_dim, 
            gnn_layer_num, 
            particle_num, 
            resamp_alpha, 
            position_size, 
            device
        ) 
        self._build_model()
    
    def _build_model(self):
        self.clip_model, _ = clip.load('ViT-B/32', device=self.device)
        self.clip_model.requires_grad_(False)
        self.preprocess = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.tokenizer = clip.tokenize
        self.obj_grounding = Visual_Object_Grounding(self.device, self.word_embd_dim, self.word_embd_dim)
        self.subj_grounding = Visual_Object_Grounding(self.device, self.word_embd_dim, self.word_embd_dim)

    def tokenize_to_device(self, x):
        x_1 = self.tokenizer(x)
        return x_1.to(self.device)
    
    def encode_text(self, x):
        return self.clip_model.encode_text(x).float().to(self.device)
    
    def encode_img(self, x):
        out = self.clip_model.encode_image(x).float().to(self.device)
        # out_ = out / out.norm(dim=-1, keepdim=True)
        return out
    
    def forward(
        self, 
        text: List, 
        bboxes: List, 
        image: Tensor,
        ):
        return NotImplementedError