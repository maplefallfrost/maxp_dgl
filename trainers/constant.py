import torch.nn as nn

from torch.optim import Adam
from adabelief_pytorch import AdaBelief

NAME_TO_LOSS = {
    "cross_entropy": nn.CrossEntropyLoss
}

NAME_TO_OPTIMIZER = {
    "adam": Adam,
    "adabelief": AdaBelief
}