from message_passing import GCNMessagePassing, GATMessagePassing, AvgMessagePassing
from models.lr import LRModel
from models.mlp_attn import MLPAttn

NAME_TO_MP = {
    "gcn": GCNMessagePassing,
    "gat": GATMessagePassing,
    "avg": AvgMessagePassing,
}

NAME_TO_MODEL = {
    "lr": LRModel,
    "mlp_attn": MLPAttn
}
