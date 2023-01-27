import torch
from torch_geometric.nn import Sequential as Seq, MessagePassing
from typing import Union, Callable, Tuple
import torch
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, PairTensor, OptPairTensor, OptTensor, Size
from torch_geometric.nn.inits import reset, zeros 

class MP(MessagePassing):
    '''
    general message passing algorithm
    h_i^k = U^k(h_i^{k-1}, \max_{j\in N(i)} M^k(h_i^{k-1}, h_j^{k-1}, e_{i,j}))
    
    '''
    def __init__(self, in_channels, out_channels, edge_f_dim, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.m = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2 + edge_f_dim, in_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_channels, in_channels)
        )
        self.u = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2, in_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_channels, out_channels)
        ) 

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_features: Tensor) -> Tensor:
        """forward message
            edge_features: Tensor 
        """
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=x, edge_features=edge_features, size=None)
        # after aggregating the message m_i^k = \max_{j\in N(i)} M^k
        x_r = x[1]
        if x_r is not None:
            # compute the message after aggregation U^k(h_i^{k-1}, m_i^k)
            out = self.u(torch.hstack([x_r, out]))
        return out

    def message(self, x_i, x_j, edge_features) -> Tensor:
        '''
        compute the message M^k(h_i^{k-1}, h_j^{k-1}, e_{i,j})
            :x_i: [E, in_channels]
            :x_j: [E, in_channels]
            :edge_features: e_ij, [E, D]
        '''
        # print(x_i.size(), x_j.size(), edge_features.size()) # Debug
        f = torch.hstack([x_i, x_j, edge_features]) # input features x_i, x_j, e_ij
        # print(inp)
        return self.m(f)

class MPNN_conv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, aggr: str = 'add',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = torch.nn.Linear(in_channels, in_channels * out_channels)
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels_l = in_channels[0]

        if root_weight:
            self.lin = Linear(in_channels[1], out_channels, bias=False,
                              weight_initializer='uniform')

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        if self.root_weight:
            self.lin.reset_parameters()
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None and self.root_weight:
            out += self.lin(x_r)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # print(self.nn)
        weight = self.nn(edge_attr)
        weight = weight.view(-1, self.in_channels_l, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr}, nn={self.nn})')

class MPNN(torch.nn.Module):
    '''message passing neural network'''
    def __init__(self, layer_num, out_channels=64, in_channels=64, edge_f_dim=64):
        super().__init__()
        model_list = []
        model_list.append((MP(in_channels=in_channels, out_channels=64, edge_f_dim=edge_f_dim), 'x, edge_index, edge_features -> x'))
        model_list.append(torch.nn.ReLU(inplace=True))
        for _ in range(layer_num):
            model_list.append((MP(in_channels=64, out_channels=64, edge_f_dim=edge_f_dim), 'x, edge_index, edge_features -> x'))
            model_list.append(torch.nn.ReLU(inplace=True))
        model_list.append(torch.nn.Linear(64, out_channels))
        self.model = Seq('x, edge_index, edge_features', model_list)
        
    def forward(self, feature, edge_index, edge_attr):
        # inputs = feature, edge_index, edge_attr
        x = self.model(feature, edge_index, edge_features=edge_attr)
        # x = self.model(inputs)
        return x

class MPNN_e(torch.nn.Module):
    '''message passing neural network'''
    def __init__(self, layer_num, out_channels=64, in_channels=64):
        super().__init__()
        model_list = []
        model_list.append((MPNN_conv(in_channels=in_channels, out_channels=64), 'x, edge_index, edge_attr -> x'))
        model_list.append(torch.nn.ReLU(inplace=True))
        for _ in range(layer_num):
            model_list.append((MPNN_conv(in_channels=64, out_channels=64), 'x, edge_index, edge_attr -> x'))
            model_list.append(torch.nn.ReLU(inplace=True))
        model_list.append(torch.nn.Linear(64, out_channels))
        self.model = Seq('x, edge_index, edge_attr', model_list)
        
    def forward(self, feature, edge_index, edge_attr):
        # inputs = feature, edge_index, edge_attr
        x = self.model(feature, edge_index, edge_attr=edge_attr)
        # x = self.model(inputs)
        return x