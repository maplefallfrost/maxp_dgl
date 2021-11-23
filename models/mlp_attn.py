from models.base import PytorchBaseModel
from typing import Dict, Any
from models.layers import create_mlp_layer

import torch
import torch.nn as nn

class MLPAttn(PytorchBaseModel):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._init_params()

    def _init_params(self) -> None:
        mp_config = self.config["message_passing"]
        nn_config = self.config["nn"]

        input_dim = self.config["input_dim"]
        hidden_dim = nn_config["hidden_dim"]
        activation = nn_config["activation"]
        mp_max_depth = mp_config["max_depth"]

        self.proj = nn.Linear(input_dim, self.config["num_classes"])

        self.mlps = nn.ModuleList([
            create_mlp_layer(input_dim, hidden_dim, input_dim, activation)
            for i in range(mp_max_depth + 1)
        ])

        if "recursive" in self.config["attn_type"]:
            self.rs_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 1),
                    nn.Sigmoid()
                )
                for i in range(mp_max_depth + 1)
            ])
        
        if "jk" in self.config["attn_type"]:
            self.jk_gates = nn.ModuleList([
                nn.Sequential(
                    create_mlp_layer(input_dim * (mp_max_depth + 1), hidden_dim, 1, activation),
                    nn.Sigmoid()
                )
                for i in range(mp_max_depth + 1)
            ])
        
    def forward(self, collate_batch: Dict[str, Any]) -> torch.Tensor:
        x = collate_batch["x"]
        input_dim = self.config["input_dim"]
        output = x[:, :self.config["input_dim"]]
        for i in range(len(self.mlps)):
            cur_x = x[:, i*input_dim:(i+1)*input_dim]
            mlp_out = self.mlps[i](cur_x)
            gates = []
            if "recursive" in self.config["attn_type"]:
                rs_gates = self.rs_gates[i](mlp_out)
                gates.append(rs_gates)
            
            if "jk" in self.config["attn_type"]:
                jk_gates = self.jk_gates[i](x)
                gates.append(jk_gates)
            
            mean_gates = torch.mean(torch.stack(gates), dim=0)
            output = mean_gates * mlp_out + (1 - mean_gates) * output
        
        logits = self.proj(output)
        return {"logits": logits, "last_emb": output}
