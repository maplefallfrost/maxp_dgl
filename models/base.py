from typing import Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path

import pickle
import numpy as np
import torch as th
import torch.nn as nn

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, config):
        raise NotImplementedError

    @abstractmethod
    def save(self, model_save_path):
        raise NotImplementedError
    
    @abstractmethod
    def load(self, model_load_path):
        raise NotImplementedError
    

class SklearnBaseModel(BaseModel):
    def prepare(self, data: Dict[str, Any]) -> None:
        feats = [data["node_feat"].numpy()]
        mp_config = self.config["message_passing"]
        for i in range(mp_config["max_depth"]):
            feats.append(data["{}_{}".format(mp_config["mode"], i + 1)])
        X = np.hstack(feats)
        y = data["labels"]

        train_index = data["train_index"]
        valid_index = data["valid_index"]
        test_index = data["test_index"]
        self.X_train, self.y_train = X[train_index, :], y[train_index]
        self.X_valid, self.y_valid = X[valid_index, :], y[valid_index]
        self.X_test, self.y_test = X[test_index, :], y[test_index]

    def fit(self) -> None:
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self, mode: str) -> np.ndarray:
        if mode == "train":
            return self.model.predict(self.X_train)
        if mode == "valid":
            return self.model.predict(self.X_valid)
        if mode == "test":
            return self.model.predict(self.X_test)
    
    def save(self, model_save_path: Path) -> None:
        with open(model_save_path, "wb") as f:
            pickle.dump(self.model, f)
    
    def load(self, model_load_path: Path) -> None:
        with open(model_load_path, "rb") as f:
            self.model = pickle.load(f)


class PytorchBaseModel(BaseModel, nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.device = th.device("cuda")

    def load(self, model_load_path: Path) -> None:
        self.load_state_dict(th.load(model_load_path))
    
    def save(self, model_save_path: Path) -> None:
        th.save(self.state_dict(), model_save_path)