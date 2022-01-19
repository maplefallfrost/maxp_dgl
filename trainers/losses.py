import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

from typing import Dict, Any
from trainers.flexmatch_utils import consistency_loss

def loge_loss(logits, gt_labels):
    sigma = 1 - math.log(2)
    loss = torch.log(sigma - torch.log(F.softmax(logits, dim=1)[:, gt_labels]))
    return loss.mean()

class LossWrapper:
    @staticmethod
    def forward(
        loss_name: str, 
        **kwargs
    ) -> torch.Tensor:

        if loss_name == "cross_entropy":
            logits = kwargs["logits"]
            gt_labels = kwargs["y"]
            loss = nn.CrossEntropyLoss()(logits, gt_labels)
        if loss_name == "loge":
            logits = kwargs["logits"]
            gt_labels = kwargs["y"]
            loss = loge_loss(logits, gt_labels)
        if loss_name == "flexmatch_consistency":
            loss = consistency_loss(**kwargs)
        
        return loss
