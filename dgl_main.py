import argparse
import fitlog
import dgl
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import sklearn

from pathlib import Path
from typing import Dict, Any
from utils import load_config, setup_seed, load_dgl_graph, add_labels, set_mask, AverageMeter
from models.model_rev import RevGAT
from models.gat import GAT
from trainers.constant import NAME_TO_OPTIMIZER
from dgl.dataloading.neighbor import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
from tqdm import tqdm
from trainers.losses import LossWrapper
from predictors import NAME_TO_PREDICTOR
from evaluate import eval_fn

fitlog.set_log_dir("logs/")

def gen_model(args):
    if args.use_labels:
        n_node_feats_ = args.n_node_feats + args.n_classes
    else:
        n_node_feats_ = args.n_node_feats

    if args.backbone == "rev":
        model = RevGAT(
                      n_node_feats_,
                      args.n_classes,
                      n_hidden=args.n_hidden,
                      n_layers=args.n_layers,
                      n_heads=args.n_heads,
                      activation=F.relu,
                      dropout=args.dropout,
                      input_drop=args.input_drop,
                      attn_drop=args.attn_drop,
                      edge_drop=args.edge_drop,
                      use_attn_dst=not args.no_attn_dst,
                      use_symmetric_norm=args.use_norm)
    elif args.backbone == "gat":
        model = GAT(
            n_node_feats_,
            0,
            args.n_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_hidden=args.n_hidden,
            edge_emb=0,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            edge_drop=args.edge_drop,
            use_attn_dst=not args.no_attn_dst
        )
    else:
        raise Exception("Unknown backnone")

    return model


def train(config, model, graph, feat, labels, train_idx, val_idx, test_idx, device):
    args = config["model"]
    cluster_config = config["cluster"]
    trainer_config = config["trainer"]

    if args.use_labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate
        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]
        feat = add_labels(feat, labels, train_labels_idx, args.n_classes)
    else:
        train_pred_idx = train_idx

    graph.ndata["id"] = torch.arange(graph.number_of_nodes())
    graph.ndata["feat"] = feat
    graph.ndata["labels"] = labels
    set_mask(graph, train_idx, "train_mask")
    set_mask(graph, val_idx, "val_mask")
    set_mask(graph, test_idx, "test_mask")
    set_mask(graph, train_pred_idx, "train_pred_mask")

    train_loader = dgl.dataloading.GraphDataLoader(
        dgl.dataloading.ClusterGCNSubgraphIterator(
            graph,
            cluster_config["num_partitions"], 
            cluster_config["cache_dir"]
        ), 
        shuffle=True,
        batch_size=trainer_config["train_batch_size"], 
        num_workers=trainer_config["num_workers"]
    )

    optimizer_config = config["optimizer"]
    optimizer = NAME_TO_OPTIMIZER[optimizer_config["name"]](model.parameters(), **optimizer_config["kwargs"])

    max_train_steps = trainer_config["max_epoch"] * len(train_loader)
    progress_bar = tqdm(range(max_train_steps))
    losses = AverageMeter()
    best_valid_metric = 0
    global_step = 0
    predictor = NAME_TO_PREDICTOR[config["predict"]["name"]](config)
    predictor.prepare(
        data={"graph": graph, "labels": labels, 
              "train_idx": train_idx, "valid_idx": val_idx, "test_idx": test_idx}
    )

    for epoch in range(trainer_config["max_epoch"]):
        model.train()
        for cluster in train_loader:
            global_step += 1
            cluster = cluster.to(device)

            batch_train_pred_masks = cluster.ndata["train_pred_mask"]
            actual_batch_size = batch_train_pred_masks.sum().item()
            if actual_batch_size == 0:
                continue
            batch_labels = cluster.ndata["labels"]
            logits = model(cluster)

            loss = 0
            for loss_name in config["losses"]:
                if loss_name == "cross_entropy":
                    loss += LossWrapper.forward(
                        loss_name,
                        logits=logits[batch_train_pred_masks],
                        y=batch_labels[batch_train_pred_masks]
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), actual_batch_size)
            if global_step % trainer_config["log_every"] == 0:
                fitlog.add_loss(losses.avg, name="loss", step=global_step)
            progress_bar.update(1)
    
        valid_metric = eval_fn(model, predictor, mode="valid")

        fitlog.add_metric({"valid": {"accuracy": valid_metric}}, step=epoch)
        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
            fitlog.add_best_metric({"valid": {"accuracy": best_valid_metric}})
            model.save(config["model_save_path"])


def run(config, graph, feat, labels, train_idx, val_idx, test_idx, device):
    model = gen_model(config["model"]).to(device)
    train(config, model, graph, feat, labels, train_idx, val_idx, test_idx, device)


def train_eval_fn(config: Dict[str, Any]) -> None:
    data = load_dgl_graph(config["data_dir"])
    graph, labels = data["graph"], data["labels"]
    feat = data["node_feat"]
    train_idx, val_idx, test_idx = data["train_index"], data["valid_index"], data["test_index"]

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(feat.numpy())
    norm_feat = scaler.transform(feat.numpy())
    feat = torch.FloatTensor(norm_feat)

    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    device = torch.device("cuda:{}".format(config["gpu"]))
    run(config, graph, feat, labels, train_idx, val_idx, test_idx, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dgl trainer for maxp dgl contest")
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument(
        "--mode", 
        choices=["train", "eval", "submit"], 
        required=True)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()
    MODE_TO_FN = {
        "train": train_eval_fn,
        "eval": train_eval_fn
    }
    config = load_config(args.config_path)
    config["gpu"] = args.gpu

    if "rng_seed" in config:
        setup_seed(config["rng_seed"])
    config["model"] = argparse.Namespace(**config["model"])
    print("mode: {}".format(args.mode))
    MODE_TO_FN[args.mode](config)
    fitlog.finish()
