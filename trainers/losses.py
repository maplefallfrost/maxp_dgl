import torch

from typing import Dict, Any

import torch.nn as nn

class LossWrapper:
    @staticmethod
    def forward(
        config: Dict[str, Any],
        collate_batch: Dict[str, Any],
        output: Dict[str, Any]
    ) -> torch.Tensor:

        logits = output["logits"]
        gt_labels = collate_batch["y"]

        loss = 0
        for loss_name in config["losses"]:
            if loss_name == "cross_entropy":
                loss += nn.CrossEntropyLoss()(logits, gt_labels)
        
        return loss
