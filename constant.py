from message_passing import GCNMessagePassing, GATMessagePassing
from models.lr import LRModel
from models.mlp_attn import MLPAttn

NAME_TO_MP = {
    "gcn": GCNMessagePassing,
    "gat": GATMessagePassing
}

NAME_TO_MODEL = {
    "lr": LRModel,
    "mlp_attn": MLPAttn
}
