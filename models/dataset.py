import numpy as np
import torch as th
from typing import List, Dict, Any

class Dataset:
    def __init__(self, X, y, subset) -> None:
        self.X = X
        self.y = y
        self.subset = subset
    
    def __getitem__(self, i) -> Dict[str, Any]:
        i = self.subset[i]
        return {
            "idx": th.LongTensor([i]),
            "x": np.array(self.X[i, :]),
            "y": np.array(self.y[i])
        }
    
    def __len__(self) -> int:
        return len(self.subset)


class LabelPropDataset:
    def __init__(self, X, prop_label, y, subset) -> None:
        self.X = X
        self.prop_label = prop_label
        self.y = y
        self.subset = subset
    
    def __getitem__(self, i) -> Dict[str, Any]:
        i = self.subset[i]
        return {
            "idx": th.LongTensor([i]),
            "x": np.array(self.X[i, :]),
            "prop_label": np.array(self.prop_label[i]),
            "y": np.array(self.y[i])
        }

    def __len__(self) -> int:
        return len(self.subset)


class Collator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = batch[0].keys()
        collate_batch = {}
        for key in keys:
            collate_data = [x[key] for x in batch]
            if isinstance(collate_data[0], np.ndarray):
                collate_data = [th.from_numpy(x) for x in collate_data]
            tensor = th.stack(collate_data)
            collate_batch[key] = tensor
        return collate_batch
