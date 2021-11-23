from sklearn.linear_model import LogisticRegression
from typing import Dict, Any
from torch.utils.data import DataLoader
from models.base import SklearnBaseModel

import numpy as np


class LRModel(SklearnBaseModel):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model = LogisticRegression(**config["model_kwargs"])
    