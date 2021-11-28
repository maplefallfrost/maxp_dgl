import numpy as np

from sklearn.metrics import accuracy_score
from predictors.base import BasePredictor

def eval_fn(model, predictor: BasePredictor, mode: str) -> float:
    preds = predictor.predict(model, mode)
    MODE_TO_GT = {
        "train": predictor.y_train,
        "valid": predictor.y_valid,
    }
    acc = accuracy_score(MODE_TO_GT[mode], preds)
    return acc
