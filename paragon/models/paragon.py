import torch
import torch.nn as nn
import clip

from typing import List, Tuple
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Normalize, Resize
# real 
from paragon.models.PGNN import PMFGNN_CNN_RS_RP, PMFGNN_CNN_Fusion
from paragon.models.paragon_base import ParaGonClipBase
from paragon.models.PGNN.pgnn import PGNN
from paragon.models.SoftParsing.softparsing import SoftParsing, gnn_par_attn
from paragon.models.VisualGrounding import Visual_Object_Grounding
from paragon.models.SoftParsing import gnn_par_attn_pos_embd, gnn_par_attn_pos_embd2
from paragon.models.SoftParsing import gnn_par_linear_pos_embd
from paragon.models.SoftParsing.dep_tree import dep_tree_parser, extract_dep_edge_info
from paragon.utils.grounding_utils import bboxes_to_pos_matrix, from_bboxes_to_image, from_bboxes_to_image_cliport, bboxes_to_coord

BICUBIC = InterpolationMode.BICUBIC

class ParaGon(ParaGonClipBase):
    def __init__(
        self, 
        aggr: str, 
        word_embd_dim: int, 
        embd_dim: int, 
        gnn_layer_num: int, 
        particle_num: int, 
        resamp_alpha: float, 
        position_size: int, 
        device: str,
        return_loss = False
        ) -> None:
        super().__init__(
            aggr, 
            word_embd_dim, 
            embd_dim, 
            gnn_layer_num, 
            particle_num, 
            resamp_alpha, 
            position_size, 
            device)
        self.return_loss= return_loss
        self._build()
    
    def _build(self):
        self.pmfgnn = PGNN(
            aggr=self.aggr,
            num_layers=self.gnn_layer_num,
            embd_dim=self.embd_dim,
            particle_num=self.particle_num,
            output_dim=2,
            resamp_alpha=self.resamp_alpha,
            position_size=self.position_size,
            word_embd_dim=self.word_embd_dim,
            device=self.device,
        )
        self.gnn_par = SoftParsing(layer_num=4, device=self.device)

    def forward(
        self, 
        text: List,
        bboxes: List,
        image: List,
        ):
        
        dep_tree = self.dep_parser(text)
        node_pos, node_phrases, edge_idx, edge_tag = \
            map(list, zip(*map(extract_dep_edge_info, dep_tree)))
        
        phrases_tokens = list(map(self.tokenize_to_device, node_phrases))
        phrases_features = list(map(self.encode_text, phrases_tokens))
        
        t_embd, c_embd, r_embd, g_embd = zip(*map(self.gnn_par, node_pos, edge_idx, edge_tag, phrases_features))

        imgs = list(map(from_bboxes_to_image, bboxes, image))
        imgs = [torch.stack(list(map(self.preprocess, img))).to(self.device) for img in imgs]
        
        pos_matrix = [bboxes_to_pos_matrix(
            size=self.position_size, 
            n_fn=self.n_fn,
            r_fn=self.r_fn,
            bboxes=bbox  
        ) for bbox in bboxes]

        coord_tensor = [bboxes_to_coord(
            bboxes=bbox
        ).to(self.device) for bbox in bboxes]
        
        visual_features = list(map(self.encode_img, imgs))

        weights, _ = self.obj_grounding(c_embd, visual_features)
        tar_weights, _ = self.subj_grounding(t_embd, visual_features)
        weights = [weight - torch.logsumexp(weight, dim=-1, keepdim=True) for weight in weights]

        p, p_w, p_h = self.pmfgnn((pos_matrix, weights), torch.stack(r_embd).to(self.device))
        placement = (p, p_w,)
        if self.return_loss:
            text_gene_loss = torch.nn.functional.mse_loss(
                torch.stack(g_embd).view(1, -1, self.word_embd_dim), torch.stack(phrases_features).view(1, -1, self.word_embd_dim)
            )
            return placement, tar_weights, coord_tensor, text_gene_loss
        else:
            return placement, tar_weights, coord_tensor, p_h