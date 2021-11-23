from typing import Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path
from models.dataset import Dataset, Collator
from torch.utils.data import DataLoader
from utils import to_device, AverageMeter
from tqdm import tqdm
from torch.optim import Adam
from eval import eval_fn

import fitlog
import pickle
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, config):
        raise NotImplementedError

    @abstractmethod
    def prepare(self, data):
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, mode):
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

    def prepare(self, data: Dict[str, Any]) -> None:
        feats = [data["node_feat"]]
        mp_config = self.config["message_passing"]
        for i in range(mp_config["max_depth"]):
            feats.append(th.from_numpy(data["{}_{}".format(mp_config["mode"], i + 1)]))
        X = th.hstack(feats)
        y = data["labels"]

        train_index = data["train_index"]
        valid_index = data["valid_index"]
        test_index = data["test_index"] 
        
        X_train, y_train = X[train_index, :], y[train_index]
        X_valid, y_valid = X[valid_index, :], y[valid_index]
        X_test, y_test = X[test_index, :], y[test_index] 
        self.y_train = y_train
        self.y_valid = y_valid

        train_dataset = Dataset(X_train, y_train)
        valid_dataset = Dataset(X_valid, y_valid)
        test_dataset = Dataset(X_test, y_test)

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            collate_fn=Collator(),
            num_workers=self.config["num_workers"]
        )

        # use for eval training data
        self.static_train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=Collator(),
            num_workers=self.config["num_workers"]
        )

        self.valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=Collator(),
            num_workers=self.config["num_workers"]
        )

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=Collator(),
            num_workers=self.config["num_workers"]
        )
    
    def fit(self) -> None:
        loss_fn = nn.CrossEntropyLoss()
        max_train_steps = self.config["max_epoch"] * len(self.train_loader)
        progress_bar = tqdm(range(max_train_steps))
        optimizer = Adam(self.parameters(), lr=self.config["lr"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["max_epoch"])
        losses = AverageMeter()
        best_valid_metric = 0
        global_step = 0
        for epoch in range(self.config["max_epoch"]):
            for collate_batch in self.train_loader:
                collate_batch = to_device(collate_batch, self.device)
                global_step += 1

                self.train()
                logits = self.forward(collate_batch)
                loss = loss_fn(logits, collate_batch["y"])
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                losses.update(loss.item(), self.config["batch_size"])
                if global_step % self.config["log_every"] == 0:
                    fitlog.add_loss(loss.item(), name="loss", step=global_step)
            
            valid_metric = eval_fn(self, mode="valid")
            fitlog.add_metric({"valid": {"accuracy": valid_metric}}, step=epoch)
            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                fitlog.add_best_metric({"valid": {"accuracy": best_valid_metric}})
                self.save(self.config["model_save_path"])


    
    def predict(self, mode: str) -> np.ndarray:
        NAME_TO_LOADER = {
            "train": self.static_train_loader,
            "valid": self.valid_loader,
            "test": self.test_loader
        }

        loader = NAME_TO_LOADER[mode]
        preds = []
        for collate_batch in loader:
            collate_batch = to_device(collate_batch, self.device)
            with th.no_grad():
                cur_logits = self.forward(collate_batch)
            _, cur_preds = th.max(cur_logits, dim=1)
            preds.append(cur_preds.detach().cpu().numpy())
        preds = np.hstack(preds)
        return preds

    
    def load(self, model_load_path: Path) -> None:
        self.load_state_dict(th.load(model_load_path))
    
    def save(self, model_save_path: Path) -> None:
        th.save(self.state_dict(), model_save_path)
