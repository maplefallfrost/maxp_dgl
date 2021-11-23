import dgl
import torch
import torch.nn as nn
import numpy as np
import dgl.function as fn

from typing import Dict, Any
from abc import ABC, abstractmethod


class MessagePassing(ABC):
    @abstractmethod
    def forward(self, graph, feat):
        raise NotImplementedError
    

class GCNMessagePassing(MessagePassing):
    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        graph.ndata["ci"] = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5)
        graph.ndata["cj"] = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
        graph.ndata["h"] = feat
        graph.update_all(self.message_func, self.reduce_func)
        return graph.ndata["h"]
    
    def message_func(self, edges):
        return {'m' : edges.src['h'], 'ci' : edges.src['ci']}

    def reduce_func(self, nodes):
        ci = nodes.mailbox['ci'].unsqueeze(2)
        newh = (nodes.mailbox['m'] * ci).sum(1) * nodes.data['cj'].unsqueeze(1)
        return {'h' : newh}


class GATMessagePassing(MessagePassing):
    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        graph.ndata["h"] = feat
        graph.apply_edges(fn.u_dot_v("h", "h", "a"))
        self.edge_softmax(graph)
        graph.update_all(fn.src_mul_edge("h", "a", "h"), fn.sum("h", "h"))
        graph.ndata["h"] = graph.ndata["h"] / graph.ndata["z"]
        return graph.ndata["h"]

    def edge_softmax(self, graph):
        # compute the max
        graph.update_all(fn.copy_edge('a', 'a'), fn.max('a', 'a_max'))
        # minus the max and exp
        graph.apply_edges(lambda edges: {'a': torch.exp(edges.data['a'] - edges.dst['a_max'])})
        # compute normalizer
        graph.update_all(fn.copy_edge('a', 'a'), fn.sum('a', 'z'))


