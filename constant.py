from message_passing import GCNMessagePassing
from models.lr import LRModel
from models.mlp_attn import MLPAttn

NAME_TO_MP = {
    "gcn": GCNMessagePassing
}

NAME_TO_MODEL = {
    "lr": LRModel,
    "mlp_attn": MLPAttn
}
