import torch
import numpy as np
import networkx as nx
from itertools import islice

def path_to_tuples(paths):
    tuple_paths = []
    for path in paths:
        tuple_path = []
        pre_ele = None
        for ele in path:
            if pre_ele is None:
                pre_ele = ele
                continue
            else:
                tuple_path.append((pre_ele, ele))
                pre_ele = ele
        tuple_paths.append(tuple_path)
    return tuple_paths

def matrix_to_path(matrix: torch.Tensor):
    shape_0 , shape_1 = matrix.size()
    path = []
    for i in range(shape_0):
        for j in range(shape_1):
            if matrix[i,j]==1:
                path.append(j)
    return [path]


def get_k_unlabeled_adj_matrix(adj, adj_origin, source, target, k, weight=None):
    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()
    shape = np.shape(adj)
    unlabeled = np.zeros(shape)
    G = nx.from_numpy_array(adj)
    paths = list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )
    # print(paths)
    paths = path_to_tuples(paths)
    for path in paths:
        for edge in path:
            assert adj_origin[edge[0], edge[1]] == 1
            unlabeled[edge[0], edge[1]] = 1.0
    return torch.tensor(unlabeled)
