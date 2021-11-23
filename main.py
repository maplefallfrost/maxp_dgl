import argparse
import fitlog
import dgl
import os
import numpy as np
import sys
import torch

from pathlib import Path
from utils import load_config, load_dgl_graph, setup_seed
from constant import NAME_TO_MP, NAME_TO_MODEL
from typing import Dict, Any
from eval import eval_fn
from models.base import PytorchBaseModel, SklearnBaseModel
from tqdm import trange


fitlog.set_log_dir("logs/")


def message_passing_fn(config: Dict[str, Any]) -> None:
    data = load_dgl_graph(config["data_dir"])
    graph = data["graph"]
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    node_feat = data["node_feat"]

    os.makedirs(config["save_dir"], exist_ok=True)

    mp = NAME_TO_MP[config["mp_mode"]]()
    h = node_feat
    for i in range(config["max_depth"]):
        cur_save_path = os.path.join(config["save_dir"], "{}_{}.npy".format(config["mp_mode"], str(i + 1)))
        if os.path.exists(cur_save_path):
            h = torch.from_numpy(np.load(cur_save_path))
            continue
        
        print("message passing count: {}".format(i + 1))
        h = mp.forward(graph, h)

        with open(cur_save_path, "wb") as f:
            np.save(f, h.detach().numpy())


def merge_mp_files(config: Dict[str, Any]) -> None:
    data = load_dgl_graph(config["data_dir"])
    feat = data["node_feat"].numpy()

    out_path = os.path.join(config["save_dir"], "{}_all.npy".format(config["mp_mode"]))

    rows = feat.shape[0]
    cols = (config["max_depth"] + 1) * feat.shape[1]
    out = np.memmap(out_path, dtype=feat.dtype, mode="w+", shape=(rows, cols))
    out[:, :feat.shape[1]] = feat
    start = feat.shape[1]
    for i in trange(0, config["max_depth"]):
        cur_mp = np.load(os.path.join(config["save_dir"], "{}_{}.npy".format(config["mp_mode"], str(i + 1))))
        out[:, start:start+feat.shape[1]] = cur_mp
        start += feat.shape[1]


def train_eval_fn(config: Dict[str, Any]) -> None:
    data = load_dgl_graph(config["data_dir"])

    if config["in_memory"]:
        mp_config = config["message_passing"]
        for i in range(mp_config["max_depth"]):
            mp_feat = np.load(os.path.join(mp_config["data_dir"], "{}_{}.npy".format(mp_config["mode"], i + 1)))
            data["{}_{}".format(mp_config["mode"], i + 1)] = mp_feat

    model = NAME_TO_MODEL[config["model_name"]](config)
    model.prepare(data)
    if isinstance(model, PytorchBaseModel):
        device = torch.device("cuda")
        model = model.to(device)

    if config["mode"] == "train":
        model.fit()
        train_acc = eval_fn(model, mode="train")
        print("train accuracy: {}".format(train_acc))
        if isinstance(model, PytorchBaseModel):
            model.load(config["model_save_path"])
    else:
        model.load(config["model_save_path"])

    valid_acc = eval_fn(model, mode="valid")
    print("valid accuracy: {}".format(valid_acc))

    if config["mode"] == "train" and isinstance(model, SklearnBaseModel):
        os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)
        model.save(config["model_save_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="maxp dgl contest")
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--mode", choices=["message_passing", "merge_mp", "train", "eval"], required=True)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    config = load_config(args.config_path)
    config["mode"] = args.mode
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    MODE_TO_FN = {
        "message_passing": message_passing_fn,
        "merge_mp": merge_mp_files,
        "train": train_eval_fn,
        "eval": train_eval_fn
    }

    if "rng_seed" in config:
        setup_seed(config["rng_seed"])

    MODE_TO_FN[args.mode](config)
    fitlog.finish()
