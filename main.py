import argparse
import fitlog
import dgl
import os
import numpy as np
import sys
import torch
import pandas as pd

from pathlib import Path
from utils import load_config, load_dgl_graph, setup_seed
from constant import NAME_TO_MP, NAME_TO_MODEL
from typing import Dict, Any
from evaluate import eval_fn
from models.base import PytorchBaseModel, SklearnBaseModel
from tqdm import trange
from pseudo_label import pseudo_label_fn
from eda import plot_feature_fn
from trainers import NAME_TO_TRAINER
from predictors import NAME_TO_PREDICTOR


fitlog.set_log_dir("logs/")


def message_passing_fn(config: Dict[str, Any]) -> None:
    data = load_dgl_graph(config["data_dir"])
    graph = data["graph"]
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    node_feat = data["node_feat"]

    os.makedirs(config["save_dir"], exist_ok=True)

    mp = NAME_TO_MP[config["mp_mode"]](config)
    h = node_feat
    for i in range(config["max_depth"]):
        cur_save_path = os.path.join(config["save_dir"], "{}_{}.npy".format(config["mp_mode"], str(i + 1)))
        if os.path.exists(cur_save_path):
            h = torch.from_numpy(np.load(cur_save_path, allow_pickle=True))
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
    if isinstance(model, PytorchBaseModel):
        device = torch.device("cuda")
        model = model.to(device)

    predict_config = config["predict"]
    predictor = NAME_TO_PREDICTOR[predict_config["name"]](config)
    predictor.prepare(data)
    if config["mode"] == "train":
        trainer = NAME_TO_TRAINER[config["trainer"]["name"]](config)
        trainer.prepare(data)
        trainer.fit(model)
        train_acc = eval_fn(model, predictor, mode="train")
        print("train accuracy: {}".format(train_acc))
        if isinstance(model, PytorchBaseModel):
            model.load(config["model_save_path"])
    else:
        model.load(config["model_save_path"])

    valid_acc = eval_fn(model, predictor, mode="valid")
    print("valid accuracy: {}".format(valid_acc))

    if config["mode"] == "train" and isinstance(model, SklearnBaseModel):
        os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)
        model.save(config["model_save_path"])


def submit(config: Dict[str, Any]) -> None:
    data = load_dgl_graph(config["data_dir"])
    df = pd.read_csv("../data/IDandLabels.csv")
    paper_id_to_node_id = dict(zip(df.paper_id, df.node_idx))
    sample_submit = pd.read_csv("../data/sample_submission_for_validation.csv")
    model = NAME_TO_MODEL[config["model_name"]](config)

    if isinstance(model, PytorchBaseModel):
        device = torch.device("cuda")
        model = model.to(device)
    
    predict_config = config["predict"]
    predictor = NAME_TO_PREDICTOR[predict_config["name"]](config)
    predictor.prepare(data)
    
    model.load(config["model_save_path"])
    valid_acc = eval_fn(model, predictor, mode="valid")
    print("valid accuracy: {}".format(valid_acc))

    paper_ids = sample_submit["id"]
    test_preds = predictor.predict(model, mode="test")
    node_id_to_model_id = {x.item() : i for i, x in enumerate(data["test_index"])}

    submit_preds = []
    for paper_id in paper_ids:
        node_id = paper_id_to_node_id[paper_id]
        model_id = node_id_to_model_id[node_id]
        cur_pred = chr(ord("A") + test_preds[model_id])
        submit_preds.append(cur_pred)
    sample_submit["label"] = submit_preds
    sample_submit.to_csv(config["submit_path"], index=False)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="maxp dgl contest")
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument(
        "--mode", 
        choices=["message_passing", "merge_mp", "train", "eval", "submit", "pseudo_label", "plot_feature"], 
        required=True)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    config = load_config(args.config_path)
    config["mode"] = args.mode
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    MODE_TO_FN = {
        "message_passing": message_passing_fn,
        "merge_mp": merge_mp_files,
        "train": train_eval_fn,
        "eval": train_eval_fn,
        "submit": submit,
        "pseudo_label": pseudo_label_fn,
        "plot_feature": plot_feature_fn,
    }

    if "rng_seed" in config:
        setup_seed(config["rng_seed"])

    print("mode: {}".format(args.mode))
    MODE_TO_FN[args.mode](config)
    fitlog.finish()
