from trainers.vanilla import VanillaPytorchTrainer 
from trainers.flexmatch import FlexMatchTrainer
from trainers.label_prop import LabelPropPytorchTrainer

NAME_TO_TRAINER = {
    "vanilla": VanillaPytorchTrainer,
    "flexmatch": FlexMatchTrainer,
    "label_prop": LabelPropPytorchTrainer
}