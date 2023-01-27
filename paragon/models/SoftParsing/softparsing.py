from paragon.SoftParsing.gnn import MPNN, MPNN_e
import torch
import torch.nn as nn
from paragon.utils.lang_utils import DEP_REL_DICT, POS_DICT_
from paragon.SoftParsing.dep_tree import dep_tree_parser, extract_dep_edge_info
from paragon.SoftParsing.slot_attention import SlotAttention
import clip
import math

class gnn_par_base(nn.Module):
    '''use gnn to learn the parsing'''
    def __init__(self, embd_dim=64, layer_num=2, slot_num=4, device='cuda:0') -> None:
        super().__init__()
        self.device=device
        self.pos_embd = torch.nn.Embedding(
            num_embeddings=len(POS_DICT_), 
            embedding_dim=embd_dim
        )
        self.position_embd = torch.nn.Embedding(
            num_embeddings=64,
            embedding_dim=embd_dim
        )
        self.dep_embd = torch.nn.Embedding(
            num_embeddings=len(DEP_REL_DICT), 
            embedding_dim=embd_dim
        )
        self.activation = torch.nn.LeakyReLU()
        self.gnn = MPNN(
            layer_num=layer_num, 
            out_channels=embd_dim, 
            in_channels=embd_dim, 
            edge_f_dim=embd_dim,
        )
        self.gnn_tar = MPNN(
            layer_num=layer_num, 
            out_channels=embd_dim, 
            in_channels=embd_dim, 
            edge_f_dim=embd_dim,
        )

        self.context_att = SlotAttention(
            num_slots=slot_num,
            dim=embd_dim,
        )
        self.context_fn = nn.Linear(embd_dim, 1)
        self.target_fn = nn.Linear(embd_dim, 1)
        self.context_fn = nn.Linear(embd_dim * 2, 1)
        self.edge_fn = nn.Linear(embd_dim * 2, 1)
        self.slot_num= slot_num

    def get_node_features(self, node_info):
        node_features = []
        idx = 0
        for info in node_info:
            node_feature = self.pos_embd(torch.tensor(info, device=self.device))
            # node_position = self.position_embd(torch.tensor(node_index[idx], device=self.device))
            node_features.append(node_feature)
            # node_positions.append(node_position)
            idx += 1
        return torch.vstack(node_features) 
    
    def get_edge_features(self, edge_info):
        edge_features = []
        for info in edge_info:
            edge_feature = self.dep_embd(torch.tensor(info, device=self.device))
            edge_features.append(edge_feature)
        return torch.vstack(edge_features)
    
    def forward(self, node_pos, edge_index, edge_tag, phrases):
        return NotImplementedError()

class gnn_par_attn(gnn_par_base):
    '''use gnn to learn the parsing'''
    def __init__(self, embd_dim=64, layer_num=2, slot_num=4, device='cuda:0') -> None:
        super().__init__(embd_dim, layer_num, slot_num, device)
        self.rel_attn = nn.MultiheadAttention(embd_dim, num_heads=8, batch_first=True)
    
    def forward(self, node_pos, edge_index, edge_tag, phrases):
        node_features = self.get_node_features(node_pos)
        edge_features = self.get_edge_features(edge_tag)
        # print(edge_features)
        
        
        h = self.gnn(
            node_features.to(self.device), 
            torch.tensor(edge_index).to(self.device), 
            edge_features.to(self.device)
            )

        hidden_features = self.activation(h) 

        target_score = self.target_fn(hidden_features)
        target_prob = torch.softmax(target_score, dim=0)
        context_prob_k, w_c_embd = self.context_att(hidden_features.unsqueeze(0))
        
        _, w_r_prob = self.rel_attn(w_c_embd, hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))

        # print(context_prob_k)
        # print(w_r_prob) 

        w_t_embd = torch.sum(phrases * target_prob, dim=0)
        w_c_embd = torch.sum(phrases * context_prob_k.squeeze().unsqueeze(-1), dim=1)
        w_r_embd = torch.sum(phrases * w_r_prob.squeeze().unsqueeze(-1), dim=1) 
        return w_t_embd, w_c_embd, w_r_embd

class gnn_par_attn_pos_embd_Bayesian(gnn_par_base):
    '''use gnn to learn the parsing'''
    def __init__(self, embd_dim=64, layer_num=2, slot_num=4, device='cuda:0') -> None:
        super().__init__(embd_dim, layer_num, slot_num, device)
        self.rel_attn = nn.MultiheadAttention(embd_dim, num_heads=8, batch_first=True)
        self.rel_activation = nn.LeakyReLU()
        self.layer_norm_ = nn.LayerNorm(embd_dim)
        self.attn_mlp = nn.Linear(embd_dim, embd_dim)
        self.rel_attn_out = nn.MultiheadAttention(embd_dim, num_heads=8, batch_first=True)
        self.positionalencoding = PositionalEncoding(embd_dim, max_len=64)

    def forward(self, node_pos, edge_index, edge_tag, phrases):
        node_features = self.get_node_features(node_pos)
        edge_features = self.get_edge_features(edge_tag)

        h = self.gnn(
            node_features.to(self.device), 
            torch.tensor(edge_index).to(self.device), 
            edge_features.to(self.device)
            )

        h_a = self.activation(h) 
        hidden_features = self.positionalencoding(h_a.unsqueeze(0)).squeeze()
        target_score = self.target_fn(hidden_features)
        target_prob = torch.softmax(target_score, dim=0)
        context_prob_k, w_c_embd = self.context_att(hidden_features.unsqueeze(0))
        
        w_r_embd_, _ = self.rel_attn(w_c_embd, hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        atten_f = self.layer_norm_(w_r_embd_)
        _, w_r_prob = self.rel_attn_out(self.attn_mlp(atten_f)
           , hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        # print(context_prob_k)
        # print(w_r_prob) 

        
        return target_prob, context_prob_k, w_r_prob, phrases

class gnn_par_attn_pos_embd2(gnn_par_base):
    '''use gnn to learn the parsing'''
    def __init__(self, embd_dim=64, layer_num=2, slot_num=4, device='cuda:0') -> None:
        super().__init__(embd_dim, layer_num, slot_num, device)
        self.rel_attn = nn.MultiheadAttention(embd_dim, num_heads=8, batch_first=True)
        self.rel_activation = nn.LeakyReLU()
        self.layer_norm_ = nn.LayerNorm(embd_dim)
        self.slot_num = slot_num
        self.attn_mlp = nn.Linear(embd_dim, embd_dim)
        self.rel_attn_out = nn.MultiheadAttention(embd_dim, num_heads=8, batch_first=True)
        self.positionalencoding = PositionalEncoding(embd_dim, max_len=64)

    def forward(self, node_pos, edge_index, edge_tag, phrases):
        node_features = self.get_node_features(node_pos)
        edge_features = self.get_edge_features(edge_tag)

        h = self.gnn(
            node_features.to(self.device), 
            torch.tensor(edge_index).to(self.device), 
            edge_features.to(self.device)
            )

        h_a = self.activation(h) 
        # h_tar = self.gnn_tar(
        #     node_features.to(self.device), 
        #     torch.tensor(edge_index).to(self.device), 
        #     edge_features.to(self.device)
        #     )

        # h_a_tar = self.activation(h_tar) 
        hidden_features = self.positionalencoding(h_a.unsqueeze(0)).squeeze()
        target_score = self.target_fn(hidden_features)
        target_prob = torch.softmax(target_score, dim=0)
        context_score = self.context_fn(hidden_features)
        context_prob = torch.softmax(context_score, dim=0)
        w_c_embd = torch.sum(hidden_features * context_prob.view(1, -1, 1), dim=1)
        w_r_embd_, _ = self.rel_attn(w_c_embd.unsqueeze(0), hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        atten_f = self.layer_norm_(w_r_embd_)
        _, w_r_prob = self.rel_attn_out(self.attn_mlp(atten_f)
           , hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        print(context_prob)
        w_t_embd = torch.sum(phrases * target_prob.view(1, -1, 1), dim=1)
        w_c_embd = torch.sum(phrases * context_prob.view(1, -1, 1), dim=1)
        w_r_embd = torch.sum(phrases * w_r_prob.view(1, -1, 1), dim=1) 
        return w_t_embd, w_c_embd, w_r_embd

class gnn_par_attn_pos_embd(gnn_par_base):
    '''use gnn to learn the parsing'''
    def __init__(self, embd_dim=64, layer_num=2, slot_num=1, device='cuda:0') -> None:
        super().__init__(embd_dim, layer_num, slot_num, device)
        self.rel_activation = nn.LeakyReLU()

        self.con_attn = nn.MultiheadAttention(embd_dim, num_heads=8, batch_first=True)
        self.rel_attn = nn.MultiheadAttention(embd_dim, num_heads=8, batch_first=True)
        self.layer_norm_ = nn.LayerNorm(embd_dim)
        self.slot_num = slot_num
        self.attn_mlp = nn.Linear(embd_dim, embd_dim)
        self.positionalencoding = PositionalEncoding(embd_dim, max_len=64)

    def forward(self, node_pos, edge_index, edge_tag, phrases):
        node_features = self.get_node_features(node_pos)
        edge_features = self.get_edge_features(edge_tag)

        h = self.gnn(
            node_features.to(self.device), 
            torch.tensor(edge_index).to(self.device), 
            edge_features.to(self.device)
            )

        h_a = self.activation(h) 
         
        hidden_features = self.positionalencoding(h_a.unsqueeze(0)).squeeze()
        target_score = self.target_fn(hidden_features)
        target_prob = torch.softmax(target_score, dim=0)
        # context_score = self.context_fn(hidden_features)
        # context_prob = torch.softmax(context_score, dim=0) 
        # w_c_embd_slot = (hidden_features * context_prob).sum(dim=0, keepdim=True)
        # print(w_c_embd_slot.size())
        w_c_embd_slot = self.context_att(hidden_features.unsqueeze(0))
        
        _, w_c_prob = self.con_attn(w_c_embd_slot, hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        _, w_r_prob = self.rel_attn(w_c_embd_slot, hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        # _, w_r_prob = self.rel_attn_out(self.attn_mlp(atten_f)
        #    , hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        # print(target_prob)
        # print(context_prob)
        w_t_embd = torch.sum(phrases * target_prob.view(1, -1, 1), dim=1)
        w_c_embd_ = torch.sum(phrases * w_c_prob.view(self.slot_num, -1, 1), dim=1)
        w_r_embd = torch.sum(phrases * w_r_prob.view(self.slot_num, -1, 1), dim=1) 
        return w_t_embd, w_c_embd_, w_r_embd

class gnn_par_linear(gnn_par_base):
    '''use gnn to learn the parsing'''
    
    def forward(self, node_pos, edge_index, edge_tag, phrases):
        node_features = self.get_node_features(node_pos)

        edge_features = self.get_edge_features(edge_tag)
        h = self.gnn(
            node_features.to(self.device), 
            torch.tensor(edge_index).to(self.device), 
            edge_features.to(self.device)
            )
        # print(hidden_features) 
        hidden_features = self.activation(h)
        target_score = self.target_fn(hidden_features)
        target_prob = torch.softmax(target_score, dim=0)
        context_prob_k, w_c_embd = self.context_att(hidden_features.unsqueeze(0))
        w_c_size = w_c_embd.size()
        weighted_context_embd_ = w_c_embd.reshape(w_c_size[1], \
            w_c_size[0], w_c_size[2]).repeat(1, len(hidden_features), 1)
        h_size = hidden_features.size()
        hidden_features_ = hidden_features.reshape(1, h_size[0], h_size[1]).repeat(w_c_size[1], 1, 1)
        # print(torch.cat([weighted_context_embd_, hidden_features_], dim=-1))
        edge_score = self.edge_fn(torch.cat([weighted_context_embd_, hidden_features_], dim=-1))
        # print(edge_score.size()) 
        # print(edge_score)
        w_r_prob = torch.softmax(edge_score, dim=1)
        # print(context_prob_k)
        # print(w_r_prob) 
        w_t_embd = torch.sum(phrases * target_prob, dim=0)
        w_c_embd = torch.sum(phrases * context_prob_k.squeeze().unsqueeze(-1), dim=1)
        w_r_embd = torch.sum(phrases * w_r_prob.squeeze().unsqueeze(-1), dim=1) 

        return w_t_embd, w_c_embd, w_r_embd

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class gnn_par_linear_pos_embd(gnn_par_base):
    '''use gnn to learn the parsing'''
    def __init__(self, embd_dim=64, layer_num=2, slot_num=4, device='cuda:0') -> None:
        super().__init__(embd_dim, layer_num, slot_num, device)
        self.positionalencoding = PositionalEncoding(embd_dim, max_len=64)

    def get_node_features(self, node_info):
        node_features = []
        idx = 0
        for info in node_info:
            node_feature = self.pos_embd(torch.tensor(info, device=self.device))
            node_features.append(node_feature)
            idx += 1
        return torch.vstack(node_features)
    
    def forward(self, node_pos, edge_index, edge_tag, phrases):
        node_features = self.get_node_features(node_pos)
        node_features = self.positionalencoding(node_features.unsqueeze(0)).squeeze()
        edge_features = self.get_edge_features(edge_tag)
        h = self.gnn(
            node_features.to(self.device), 
            torch.tensor(edge_index).to(self.device), 
            edge_features.to(self.device)
            )
        # print(hidden_features)
        
        hidden_features = self.activation(h)
        
        target_score = self.target_fn(hidden_features)
        target_prob = torch.softmax(target_score, dim=0)
        context_prob_k, w_c_embd = self.context_att(hidden_features.unsqueeze(0))
        w_c_size = w_c_embd.size()
        weighted_context_embd_ = w_c_embd.reshape(w_c_size[1], \
            w_c_size[0], w_c_size[2]).repeat(1, len(hidden_features), 1)
        h_size = hidden_features.size()
        hidden_features_ = hidden_features.reshape(1, h_size[0], h_size[1]).repeat(w_c_size[1], 1, 1)
        # print(torch.cat([weighted_context_embd_, hidden_features_], dim=-1))
        edge_score = self.edge_fn(torch.cat([weighted_context_embd_, hidden_features_], dim=-1))
        w_r_prob = torch.softmax(edge_score, dim=1)
        # print(context_prob_k)
        # print(w_r_prob) 
        w_t_embd = torch.sum(phrases * target_prob, dim=0)
        w_c_embd = torch.sum(phrases * context_prob_k.squeeze().unsqueeze(-1), dim=1)
        w_r_embd = torch.sum(phrases * w_r_prob.squeeze().unsqueeze(-1), dim=1) 

        return w_t_embd, w_c_embd, w_r_embd

class gnn_par_attn_position_embd_mpnn(gnn_par_attn_pos_embd):
    '''use gnn to learn the parsing'''
    def __init__(self, embd_dim=64, layer_num=2, slot_num=4, device='cuda:0') -> None:
        super().__init__(embd_dim, layer_num, slot_num, device)
        self.positionalencoding = PositionalEncoding(embd_dim, max_len=64)
        self.gnn = MPNN_e(
            layer_num=layer_num, 
            out_channels=embd_dim, 
            in_channels=embd_dim, 
        ) 

class CNNTextGeneration(nn.Module):
    def __init__(self, word_embd, device) -> None:
        super().__init__()
        self.word_embd = word_embd
        self.fn = nn.Linear(word_embd, word_embd)
        # self.model = nn.Sequential(
        #     nn.Conv1d(word_embd, word_embd, 3, padding=1),
        #     nn.Conv1d(word_embd, word_embd, 3, padding=1),
        #     nn.Conv1d(word_embd, word_embd, 3, padding=1),
        # )
    
    def forward(self, x):
        # sl = x.size(1)
        # hidden_feature = self.model(x.reshape(-1, self.word_embd, sl))
        out = self.fn(x) 
        return out

class LSTMTextGeneration(nn.Module):
    def __init__(self, out_embd_dim, device, lstm_size=512):
        super().__init__()
        self.lstm_size = lstm_size
        self.device=device
        self.embedding_dim = 128
        self.num_layers = 1
        self.out_embd_dim = out_embd_dim
        # n_vocab = len(dataset.uniq_words)
        # self.embedding = nn.Embedding(
        #     num_embeddings=n_vocab,
        #     embedding_dim=self.embedding_dim,
        # )
        self.lstm = nn.GRU(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            # dropout=0.2,
            bias=False,
            batch_first=True,
        )
        self.fc = nn.Linear(self.lstm_size, out_embd_dim)

    def forward(self, x):
        sl = x.size(1)
        # prev_state = self.init_state(sl)
        output, state = self.lstm(x)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size, device=self.device),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size, device=self.device))

class SoftParsingCNN(gnn_par_base):
    '''
    Soft Parsing Module
    Use Slot Attention to detect the noun phrases
    
    '''
    def __init__(self, embd_dim=64, layer_num=2, slot_num=4, device='cuda:0') -> None:
        super().__init__(embd_dim, layer_num, slot_num, device)
        self.rel_activation = nn.LeakyReLU()

        self.con_attn = nn.MultiheadAttention(embd_dim, num_heads=8, batch_first=True)
        self.rel_attn = nn.MultiheadAttention(embd_dim, num_heads=8, batch_first=True)
        self.layer_norm_ = nn.LayerNorm(embd_dim)
        self.attn_mlp = nn.Linear(embd_dim, embd_dim)
        self.positionalencoding = PositionalEncoding(embd_dim, max_len=64)
        self.cnn_decoder = LSTMTextGeneration(512, device=self.device).to(self.device)

    def forward(self, node_pos, edge_index, edge_tag, phrases):
        node_features = self.get_node_features(node_pos)
        edge_features = self.get_edge_features(edge_tag)

        h = self.gnn(
            node_features.to(self.device), 
            torch.tensor(edge_index).to(self.device), 
            edge_features.to(self.device)
            )

        h_a = self.activation(h) 
         
        hidden_features = self.positionalencoding(h_a.unsqueeze(0)).squeeze()
        target_score = self.target_fn(hidden_features)
        target_prob = torch.softmax(target_score, dim=0)
        # context_score = self.context_fn(hidden_features)
        # context_prob = torch.softmax(context_score, dim=0) 
        # w_c_embd_slot = (hidden_features * context_prob).sum(dim=0, keepdim=True)
        # print(w_c_embd_slot.size())
        w_c_embd_slot = self.context_att(hidden_features.unsqueeze(0))
        
        _, w_c_prob = self.con_attn(w_c_embd_slot, hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        _, w_r_prob = self.rel_attn(w_c_embd_slot, hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        # _, w_r_prob = self.rel_attn_out(self.attn_mlp(atten_f)
        #    , hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        # print(target_prob)
        # print(context_prob)
        # print(w_c_prob)
        # print(w_r_prob)
        w_t_embd = torch.sum(phrases * target_prob.view(1, -1, 1), dim=1)
        w_c_embd_ = torch.sum(phrases * w_c_prob.view(self.slot_num, -1, 1), dim=1)
        w_r_embd = torch.sum(phrases * w_r_prob.view(self.slot_num, -1, 1), dim=1) 
        rnn_input = torch.zeros_like(phrases, dtype=torch.float).unsqueeze(0)
        # print((w_t_embd * target_prob.view(1, -1, 1)).size())
        # print((w_c_embd_.unsqueeze(1) * w_c_prob.view(self.slot_num, -1, 1)).size())
        # print(rnn_input.size())
        rnn_input = rnn_input + w_t_embd * target_prob.view(1, -1, 1)\
            + (w_c_embd_.unsqueeze(1) * w_c_prob.view(self.slot_num, -1, 1)).sum(dim=0, keepdim=True)\
            + (w_r_embd.unsqueeze(1) * w_r_prob.view(self.slot_num, -1, 1)).sum(dim=0, keepdim=True)
        generated_text_embeddings, _ = self.cnn_decoder(rnn_input)
        return w_t_embd, w_c_embd_, w_r_embd, generated_text_embeddings

class SoftParsing(gnn_par_base):
    '''
    Soft Parsing Module
    Use Slot Attention to detect the noun phrases
    
    '''
    def __init__(self, embd_dim=64, layer_num=2, slot_num=4, device='cuda:0') -> None:
        super().__init__(embd_dim, layer_num, slot_num, device)
        self.rel_activation = nn.LeakyReLU()

        self.con_attn = nn.MultiheadAttention(embd_dim, num_heads=8, batch_first=True)
        self.rel_attn = nn.MultiheadAttention(embd_dim, num_heads=8, batch_first=True)
        self.layer_norm_ = nn.LayerNorm(embd_dim)
        self.attn_mlp = nn.Linear(embd_dim, embd_dim)
        self.positionalencoding = PositionalEncoding(embd_dim, max_len=64)
        self.cnn_decoder = LSTMTextGeneration(512, device=self.device).to(self.device)

    def forward(self, node_pos, edge_index, edge_tag, phrases):
        node_features = self.get_node_features(node_pos)
        edge_features = self.get_edge_features(edge_tag)

        h = self.gnn(
            node_features.to(self.device), 
            torch.tensor(edge_index).to(self.device), 
            edge_features.to(self.device)
            )

        h_a = self.activation(h) 
         
        hidden_features = self.positionalencoding(h_a.unsqueeze(0)).squeeze()
        target_score = self.target_fn(hidden_features)
        target_prob = torch.softmax(target_score, dim=0)
        # context_score = self.context_fn(hidden_features)
        # context_prob = torch.softmax(context_score, dim=0) 
        # w_c_embd_slot = (hidden_features * context_prob).sum(dim=0, keepdim=True)
        # print(w_c_embd_slot.size())
        w_c_embd_slot = self.context_att(hidden_features.unsqueeze(0))
        
        w_c_s_embd, w_c_prob = self.con_attn(w_c_embd_slot, hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        _, w_r_prob = self.rel_attn(w_c_s_embd, hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        # _, w_r_prob = self.rel_attn_out(self.attn_mlp(atten_f)
        #    , hidden_features.unsqueeze(0), hidden_features.unsqueeze(0))
        # print(target_prob)
        # print(context_prob)
        # print(w_c_prob)
        # print(w_r_prob)
        w_t_embd = torch.sum(phrases * target_prob.view(1, -1, 1), dim=1)
        w_c_embd_ = torch.sum(phrases * w_c_prob.view(self.slot_num, -1, 1), dim=1)
        w_r_embd = torch.sum(phrases * w_r_prob.view(self.slot_num, -1, 1), dim=1) 
        rnn_input = torch.zeros_like(phrases, dtype=torch.float).unsqueeze(0)
        # print((w_t_embd * target_prob.view(1, -1, 1)).size())
        # print((w_c_embd_.unsqueeze(1) * w_c_prob.view(self.slot_num, -1, 1)).size())
        # print(rnn_input.size())
        rnn_input = rnn_input + w_t_embd * target_prob.view(1, -1, 1)\
            + (w_c_embd_.unsqueeze(1) * w_c_prob.view(self.slot_num, -1, 1)).sum(dim=0, keepdim=True)\
            + (w_r_embd.unsqueeze(1) * w_r_prob.view(self.slot_num, -1, 1)).sum(dim=0, keepdim=True)
        generated_text_embeddings, _ = self.cnn_decoder(rnn_input)
        return w_t_embd, w_c_embd_, w_r_embd, generated_text_embeddings