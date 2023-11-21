import dataclasses
from typing import Dict

import networkx as nx
import numpy as np
import graph_tool as gt

from .graph_funcs import directed_extended_barabasi_albert_graph, gnp_random_graph, binomial_tree


@dataclasses.dataclass
class ModelFactory:
    @classmethod
    def directed_extended_barabasi_albert_graph(cls):
        return directed_extended_barabasi_albert_graph

    @classmethod
    def gnp_random_graph(cls):
        return gnp_random_graph
    
    @classmethod
    def binomial_tree(cls):
        return binomial_tree

    @classmethod
    def from_str(cls, model: str, model_kwargs: Dict):
        model = getattr(cls, model)()
        G = model(**model_kwargs)
        return G


def nx2gt(G: nx.Graph):
    adj_matrix = nx.convert_matrix.to_numpy_matrix(G)
    gtG = gt.Graph(directed=True)
    gtG.add_edge_list(np.transpose(np.transpose(adj_matrix).nonzero()))
    return gtG
