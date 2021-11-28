import torch.nn as nn

NAME_TO_ACTIVATION = {
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(0.2),
    "prelu": nn.PReLU(),
    "swish": nn.SiLU()
}

def create_mlp_layer(in_dim: int, hidden_dim: int, out_dim: int, activation_name: str) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        NAME_TO_ACTIVATION[activation_name],
        nn.Linear(hidden_dim, out_dim)
    )