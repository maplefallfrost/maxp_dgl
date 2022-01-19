import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import umap

from pathlib import Path
from constant import NAME_TO_MODEL
from typing import Dict, Any
from utils import load_dgl_graph, load_config
from sklearn.manifold import TSNE

def visualize_reduction_embed(reduce_emb: np.ndarray, masks: np.ndarray, save_path: Path) -> None:
    num_color = len(np.unique(masks))
    palette = np.array(sns.color_palette("hls", num_color))

    fig = plt.figure()
    ax = plt.subplot(aspect="equal")
    for i in range(num_color):
        cur_x = reduce_emb[masks == i]
        cur_colors = masks[masks == i]
        sc = ax.scatter(cur_x[:, 0], cur_x[:, 1], c=palette[cur_colors])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        
    ax.axis("off")
    ax.axis("tight")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def visualize_labels(reduce_emb: np.ndarray, labels: np.ndarray, masks: np.ndarray, save_path: Path) -> None:
    num_classes = len(np.unique(labels)) - 1
    colors = ["b", "r"]
    for c in range(num_classes):
        fig = plt.figure()
        ax = plt.subplot(aspect="equal")
        for i in range(2):
            cur_index = np.where(np.logical_and(labels == c, masks == i))
            cur_x = reduce_emb[cur_index]
            cur_y = labels[cur_index]
            sc = ax.scatter(cur_x[:, 0], cur_x[:, 1], c=colors[i])
            plt.xlim(-25, 25)
            plt.ylim(-25, 25)
            ax.axis("off")
            ax.axis("tight")
    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + f"_{str(c)}.png")
        plt.close(fig)


def plot_feature_fn(plot_config: Dict[str, Any]) -> None:
    config = load_config(plot_config["model_config"])
    data = load_dgl_graph(config["data_dir"])

    model = NAME_TO_MODEL[config["model_name"]](config)
    model.prepare(data)

    device = torch.device("cuda")
    model = model.to(device)

    train_embs = model.get_embedding(mode="train")
    valid_embs = model.get_embedding(mode="valid")
    test_embs = model.get_embedding(mode="test")
    train_masks = np.zeros(shape=(train_embs.shape[0]), dtype=np.int32)
    valid_masks = np.ones(shape=(valid_embs.shape[0]), dtype=np.int32)
    test_masks = 2 * np.ones(shape=(test_embs.shape[0]), dtype=np.int32)

    all_embs = np.vstack([train_embs, valid_embs, test_embs])
    all_masks = np.hstack([train_masks, valid_masks, test_masks])
    all_labels = np.hstack([model.y_train, model.y_valid, model.y_test])

    NAME_TO_REDUCER = {
        "tsne": TSNE,
        "umap": umap.UMAP
    }
    if os.path.exists(plot_config["emb_save_path"]):
        reduce_emb = np.load(plot_config["emb_save_path"])
    else:
        os.makedirs(os.path.dirname(plot_config["emb_save_path"]), exist_ok=True)
        reducer = NAME_TO_REDUCER[plot_config["reducer"]]()
        reduce_emb = reducer.fit_transform(all_embs)
        np.save(plot_config["emb_save_path"], reduce_emb)

    # visualize_reduction_embed(reduce_emb, all_masks, plot_config["plot_save_path"])
    visualize_labels(reduce_emb, all_labels, all_masks, plot_config["plot_label_save_path"])
