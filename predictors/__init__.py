from predictors.vanilla import VanillaPytorchPredictor
from predictors.label_prop import LabelPropPytorchPredictor
from predictors.cluster_gcn import ClusterGCNPredictor

NAME_TO_PREDICTOR = {
    "vanilla": VanillaPytorchPredictor,
    "label_prop": LabelPropPytorchPredictor,
    "cluster_gcn": ClusterGCNPredictor
}