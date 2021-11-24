from typing import Dict, Any
from utils import load_dgl_graph, load_config, to_device
from models.dataset import Dataset, Collator
from torch.utils.data import DataLoader
from constant import NAME_TO_MODEL
from tqdm import tqdm
from eval import eval_fn

import torch
import torch.nn as nn
import numpy as np
import os

def threshold_pl(pl_config: Dict[str, Any]) -> None:
    config = load_config(pl_config["model_config_path"])
    data = load_dgl_graph(config["data_dir"])

    model = NAME_TO_MODEL[config["model_name"]](config)
    device = torch.device("cuda")
    model = model.to(device)
    model.load(config["model_save_path"])

    model.prepare(data)
    valid_acc = eval_fn(model, mode="valid")
    print("valid accuracy: {}".format(valid_acc))

    num_nodes = data["labels"].shape[0]

    total_index = set(np.arange(num_nodes).tolist())
    train_index = set(data["train_index"].numpy().tolist())
    valid_index = set(data["valid_index"].numpy().tolist())
    test_index = set(data["test_index"].numpy().tolist())

    ulb_index = total_index - train_index - valid_index - test_index
    ulb_index = sorted(list(ulb_index))

    y = data["labels"]
    X_ulb, y_ulb = model.X[ulb_index], y[ulb_index]
    ulb_dataset = Dataset(X_ulb, y_ulb)
    ulb_loader = DataLoader(
        dataset=ulb_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=Collator(),
        num_workers=config["num_workers"]
    )

    start = 0
    pl_index, pl_labels = [], []
    progress_bar = tqdm(range(len(ulb_loader)))
    for collate_batch in ulb_loader:
        collate_batch = to_device(collate_batch, device)
        with torch.no_grad():
            logits = model(collate_batch)["logits"]
        batch_size = logits.size(0)
        probs = nn.Softmax(dim=1)(logits)
        max_probs, max_index = torch.max(probs, dim=1)
        max_probs = max_probs.cpu().numpy()
        max_index = max_index.cpu().numpy()
        for i, prob in enumerate(max_probs):
            if prob >= pl_config["threshold"]:
                pl_index.append(start + i)
                pl_labels.append(max_index[i])
        start += batch_size
        progress_bar.update(1)
    
    pl_index = np.array(pl_index, dtype=np.int32)
    pl_labels = np.array(pl_labels, dtype=np.int32)
    save_dict = {"pl_index": pl_index, "pl_labels": pl_labels}
    os.makedirs(os.path.dirname(pl_config["pl_save_path"]), exist_ok=True)
    np.save(pl_config["pl_save_path"], save_dict)


def pseudo_label_fn(pl_config: Dict[str, Any]) -> None:
    NAME_TO_PL = {
        "threshold": threshold_pl
    }

    NAME_TO_PL[pl_config["pl_mode"]](pl_config)
