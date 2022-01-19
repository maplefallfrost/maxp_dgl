from models.base import PytorchBaseModel
from typing import Dict, Any
from models.layers import create_mlp_layer, NAME_TO_ACTIVATION

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

        self.input_dim = input_dim

        self.activation_fn = NAME_TO_ACTIVATION[activation]
        self.norm = nn.LayerNorm(hidden_dim)

        self.proj = nn.Linear(hidden_dim, self.config["num_classes"])

        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation_fn
        )

        mlp_input_dim = input_dim
        if self.config["label_prop"]:
            self.label_mlp = create_mlp_layer(self.config["num_classes"], hidden_dim, input_dim, activation)
            mlp_input_dim += input_dim

        self.mlps = nn.ModuleList([
            create_mlp_layer(mlp_input_dim, hidden_dim, hidden_dim, activation)
            for i in range(mp_max_depth + 1)
        ])

        if "recursive" in self.config["attn_type"]:
            self.rs_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
                for i in range(mp_max_depth + 1)
            ])
        
        if "jk" in self.config["attn_type"]:
            self.jk_gates = nn.ModuleList([
                nn.Sequential(
                    create_mlp_layer(mlp_input_dim * (mp_max_depth + 1), hidden_dim, 1, activation),
                    nn.Sigmoid()
                )
                for i in range(mp_max_depth + 1)
            ])
        
        self.dropout = nn.Dropout(nn_config["dropout"])
        self.input_dropout = nn.Dropout(nn_config["input_dropout"])
        
    def forward(self, collate_batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = collate_batch["x"]
        x = self.input_dropout(x)
        chunk_x = torch.chunk(x, len(self.mlps), dim=1)
        if self.config["label_prop"]:
            prop_label = collate_batch["prop_label"]
            chunk_prop_label = torch.chunk(prop_label, len(self.mlps), dim=1)
            label_emb = torch.cat([self.label_mlp(chunk_prop_label[i]) for i in range(len(chunk_prop_label))], dim=1)

        if self.config["reverse_order"]:
            chunk_x = chunk_x[::-1]
            if self.config["label_prop"]:
                chunk_prop_label = chunk_prop_label[::-1]

        output = self.node_encoder(chunk_x[0])
        outputs = []
        for i in range(len(self.mlps)):
            cur_x = chunk_x[i]
            if self.config["label_prop"]:
                cur_label_emb = self.label_mlp(chunk_prop_label[i])
                cur_x = torch.cat([cur_x, cur_label_emb], dim=1)
            mlp_out = self.mlps[i](cur_x)
            gates = []
            if "recursive" in self.config["attn_type"]:
                rs_gates = self.rs_gates[i](mlp_out)
                gates.append(rs_gates)
            
            if "jk" in self.config["attn_type"]:
                if self.config["label_prop"]:
                    jk_input = torch.cat([x, label_emb], dim=1)
                else:
                    jk_input = x
                jk_gates = self.jk_gates[i](jk_input)
                gates.append(jk_gates)
            
            mean_gates = torch.mean(torch.stack(gates), dim=0)
            output = mean_gates * mlp_out + (1 - mean_gates) * output
            output = self.activation_fn(self.norm(output))
            if i + 1 != len(self.mlps):
                output = self.dropout(output)
            outputs.append(output)
        
        logits = self.proj(output)
        return {"logits": logits, "last_emb": output}
