import dgl
import os
import numpy as np
import torch as th
import yaml
import re
import pickle
import random


from pathlib import Path
from typing import Dict, Any


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


def setup_seed(seed):
    """
    seed: int
    """
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True


def load_config(config_path: Path) -> Dict[str, Any]:
    # This script is to solve the bug of loading scientific notation(e.g. 5e-5) as str in yaml
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_path, "r") as fp:
        config = yaml.load(fp, Loader=loader)
    return config


def load_dgl_graph(base_path: Path) -> Dict[str, Any]:
    """
    读取预处理的Graph，Feature和Label文件，并构建相应的数据供训练代码使用。

    :param base_path:
    :return:
    """
    graphs, _ = dgl.load_graphs(os.path.join(base_path, 'graph.bin'))
    graph = graphs[0]
    print('################ Graph info: ###############')
    print(graph)

    with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
        label_data = pickle.load(f)

    labels = th.from_numpy(label_data['label'])
    tr_label_idx = th.from_numpy(label_data['tr_label_idx']).long()
    val_label_idx = th.from_numpy(label_data['val_label_idx']).long()
    test_label_idx = th.from_numpy(label_data['test_label_idx']).long()
    print('################ Label info: ################')
    print('Total labels (including not labeled): {}'.format(labels.shape[0]))
    print('               Training label number: {}'.format(tr_label_idx.shape[0]))
    print('             Validation label number: {}'.format(val_label_idx.shape[0]))
    print('                   Test label number: {}'.format(test_label_idx.shape[0]))

    # get node features
    features = np.load(os.path.join(base_path, 'features.npy'))
    node_feat = th.from_numpy(features).float()
    print('################ Feature info: ###############')
    print('Node\'s feature shape:{}'.format(node_feat.shape))

    return {
        "graph": graph,
        "labels": labels,
        "train_index": tr_label_idx, 
        "valid_index": val_label_idx, 
        "test_index": test_label_idx, 
        "node_feat": node_feat
    }


def to_device(collate_batch, device):
    device_batch = {}
    for key in collate_batch:
        if isinstance(collate_batch[key], th.Tensor):
            device_batch[key] = collate_batch[key].to(device)
        else:
            device_batch[key] = collate_batch[key]
    return device_batch
