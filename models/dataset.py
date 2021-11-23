import numpy as np
import torch as th
from typing import List, Dict, Any

class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
    
    def __getitem__(self, i) -> Dict[str, Any]:
        return {
            "x": self.X[i, :],
            "y": self.y[i]
        }
    
    def __len__(self) -> int:
        return self.y.shape[0]


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
