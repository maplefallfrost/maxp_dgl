import os
import fitlog
import numpy as np
import torch.optim as optim

from tqdm import tqdm
from typing import Dict, Any
from models.base import PytorchBaseModel
from models.dataset import Dataset, Collator
from trainers.base import BaseTrainer
from trainers.constant import NAME_TO_OPTIMIZER
from trainers.losses import LossWrapper
from predictors import NAME_TO_PREDICTOR
from adabelief_pytorch import AdaBelief
from utils import AverageMeter, to_device
from evaluate import eval_fn
from torch.utils.data import DataLoader

class VanillaPytorchTrainer(BaseTrainer):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    def prepare(self, data: Dict[str, Any]) -> None:
        feat = data["node_feat"]
        mp_config = self.config["message_passing"]
        if self.config["in_memory"]:
            feats = [feat]
            for i in range(mp_config["max_depth"]):
                feats.append(th.from_numpy(data["{}_{}".format(mp_config["mode"], i + 1)]))
            X = th.hstack(feats)
        else:
            rows = feat.shape[0]
            cols = feat.shape[1] * (mp_config["max_depth"] + 1)
            X = np.memmap(os.path.join(mp_config["data_dir"], "{}_all.npy".format(mp_config["mode"])), mode="r", dtype=np.float32, shape=(rows, cols))

        self.X = X
        y = data["labels"]

        train_index = data["train_index"]
        X_train, y_train = X[train_index, :], y[train_index]
        train_dataset = Dataset(X_train, y_train)

        self.trainer_config = self.config["trainer"]
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.trainer_config["batch_size"],
            shuffle=True,
            collate_fn=Collator(),
            num_workers=self.trainer_config["num_workers"]
        )

        predict_config = self.config["predict"]
        predictor = NAME_TO_PREDICTOR[predict_config["name"]](self.config)
        predictor.prepare(data)
        self.predictor = predictor
    
    def fit(self, model: PytorchBaseModel) -> None:
        optimizer_config = self.config["optimizer"]
        optimizer = NAME_TO_OPTIMIZER[optimizer_config["name"]](model.parameters(), **optimizer_config["kwargs"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer_config["max_epoch"])

        max_train_steps = self.trainer_config["max_epoch"] * len(self.train_loader)
        progress_bar = tqdm(range(max_train_steps))
        losses = AverageMeter()
        best_valid_metric = 0
        global_step = 0
        for epoch in range(self.trainer_config["max_epoch"]):
            model.train()
            for collate_batch in self.train_loader:
                collate_batch = to_device(collate_batch, model.device)
                global_step += 1

                optimizer.zero_grad()
                output = model(collate_batch)
                loss = LossWrapper.forward(self.config, collate_batch, output)
                loss.backward()
                optimizer.step()
                scheduler.step()

                losses.update(loss.item(), self.trainer_config["batch_size"])
                if global_step % self.trainer_config["log_every"] == 0:
                    fitlog.add_loss(loss.item(), name="loss", step=global_step)
                progress_bar.update(1)
            
            valid_metric = eval_fn(model, self.predictor, mode="valid")
            fitlog.add_metric({"valid": {"accuracy": valid_metric}}, step=epoch)
            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                fitlog.add_best_metric({"valid": {"accuracy": best_valid_metric}})
                model.save(self.config["model_save_path"])
    