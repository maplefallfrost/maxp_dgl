import dgl
import torch
import torch.nn as nn
import numpy as np
import dgl.function as fn

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class MessagePassing(ABC):
    @abstractmethod
    def __init__(self, config):
        raise NotImplementedError
    @abstractmethod
    def forward(self, graph, feat):
        raise NotImplementedError
    

class GCNMessagePassing(MessagePassing):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        graph.ndata["h"] = feat
        graph.ndata["ci"] = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5).unsqueeze(1)
        graph.ndata["cj"] = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(1)
        graph.apply_edges(fn.u_mul_v("ci", "cj", "norm"))
        graph.update_all(fn.src_mul_edge("h", "norm", "m"), fn.sum("m", "h"))
        return graph.ndata.pop("h")


class GATMessagePassing(MessagePassing):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

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


class AvgMessagePassing(MessagePassing):
    def __init__(self, config):
        self.config = config

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        graph.ndata["h"] = feat
        graph.update_all(fn.copy_src(src="h", out="m"), fn.mean(msg="m", out="h"))
        rst = graph.ndata["h"]
        if self.config["residual"]:
            rst += feat
        return rst


class ClusterGCNMessagePassing(MessagePassing):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def propagate(self, graph: dgl.DGLGraph) -> torch.Tensor:
        graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
        return graph.ndata.pop("h")

    def forward(self, graph: dgl.DGLGraph, device) -> torch.Tensor:
        loader = dgl.dataloading.GraphDataLoader(
            dgl.dataloading.ClusterGCNSubgraphIterator(
                graph,
                self.config["num_partitions"],
                self.config["cache_dir"],
            ),
            shuffle=False,
            batch_size=self.config["batch_size"]
        )
        cluster_outputs, ids = [], []
        for cluster in loader:
            cluster = cluster.to(device)
            output = self.propagate(cluster)
            cluster_outputs.append(output)
            ids.append(cluster.ndata["id"])
        cluster_outputs = torch.vstack(cluster_outputs)
        ids = torch.hstack(ids)
        outputs = torch.zeros_like(cluster_outputs)
        outputs[ids] = cluster_outputs
        return outputs


class ClusterGCNLabelProp:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def forward(
        self,
        graph: dgl.DGLGraph, 
        onehot_labels: torch.Tensor, 
        has_label_index: torch.Tensor
    ) -> np.ndarray:

        device = torch.device("cuda")
        cluster_mp = ClusterGCNMessagePassing(self.config["cluster"])
        prop_labels = [onehot_labels]

        onehot_labels = onehot_labels.to(device)
        has_label_index = has_label_index.to(device)
        for i in range(self.config["max_depth"]):
            prop_label = cluster_mp.forward(graph, device)
            normalizer = torch.sum(prop_label, dim=1, keepdim=True)
            normalizer[normalizer == 0] = 1
            prop_label /= normalizer
            prop_label[has_label_index] = onehot_labels[has_label_index]
            prop_label = prop_label.cpu()
            graph.ndata["h"] = prop_label
            prop_labels.append(prop_label.numpy())
        
        prop_labels = np.hstack(prop_labels)
        return prop_labels
