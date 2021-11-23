import dgl
import torch
import torch.nn as nn
import numpy as np

from typing import Dict, Any
from abc import ABC, abstractmethod


class MessagePassing(ABC):
    @abstractmethod
    def forward(self, graph, feat):
        raise NotImplementedError
    
    @abstractmethod
    def message_func(self, edges):
        raise NotImplementedError
    
    @abstractmethod
    def reduce_func(self, edges):
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


def GATMessagePassing(MessagePassing):
    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        graph.ndata["h"] = feat
        graph.update_all(self.message_func, self.reduce_func)
        return graph.ndata["h"]

    def message_func(self, edges):
        return 
