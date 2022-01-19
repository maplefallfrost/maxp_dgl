import os
import numpy as np
import torch as th

from typing import Dict, Any
from torch.utils.data import DataLoader
from predictors.base import BasePredictor
from models.dataset import Dataset, Collator
from utils import to_device

class VanillaPytorchPredictor(BasePredictor):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def prepare(self, data: Dict[str, Any]) -> None:
        feat = data["node_feat"]
        mp_config = self.config["message_passing"]
        rows = feat.shape[0]
        cols = feat.shape[1] * (mp_config["max_depth"] + 1)
        X = np.memmap(os.path.join(mp_config["data_dir"], "{}_all.npy".format(mp_config["mode"])), mode="r", dtype=np.float32, shape=(rows, cols))
        y = data["labels"]
        self.X = X

        train_index = data["train_index"]
        valid_index = data["valid_index"]
        test_index = data["test_index"] 

        # for used in evaluation
        self.y_train = y[train_index]
        self.y_valid = y[valid_index]

        train_dataset = Dataset(X, y, train_index)
        valid_dataset = Dataset(X, y, valid_index)
        test_dataset = Dataset(X, y, test_index)

        trainer_config = self.config["trainer"]
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=trainer_config["batch_size"],
            shuffle=False,
            collate_fn=Collator(),
            num_workers=trainer_config["num_workers"]
        )

        self.valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=trainer_config["batch_size"],
            shuffle=False,
            collate_fn=Collator(),
            num_workers=trainer_config["num_workers"]
        )

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=trainer_config["batch_size"],
            shuffle=False,
            collate_fn=Collator(),
            num_workers=trainer_config["num_workers"]
        )

        self.name_to_loader = {
            "train": self.train_loader,
            "valid": self.valid_loader,
            "test": self.test_loader
        }
    
    def predict(self, model, mode: str) -> np.ndarray:
        loader = self.name_to_loader[mode]
        model.eval()

        preds = []
        for collate_batch in loader:
            collate_batch = to_device(collate_batch, model.device)
            with th.no_grad():
                output = model(collate_batch)
            cur_logits = output["logits"]
            _, cur_preds = th.max(cur_logits, dim=1)
            preds.append(cur_preds.detach().cpu().numpy())
        return np.hstack(preds) 

    def get_embedding(self, model, mode: str) -> np.ndarray:
        loader = self.NAME_TO_LOADER[mode]
        model.eval()

        last_embs = []
        for collate_batch in tqdm(loader):
            collate_batch = to_device(collate_batch, self.device)
            with th.no_grad():
                output = model(collate_batch)
            last_embs.append(output["last_emb"].detach().cpu().numpy())
        last_embs = np.vstack(last_embs)
        return last_embs
