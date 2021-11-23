import numpy as np
from sklearn.metrics import accuracy_score

def eval_fn(model, mode: str) -> float:
    model.eval()
    preds = model.predict(mode)
    MODE_TO_GT = {
        "train": model.y_train,
        "valid": model.y_valid,
    }
    acc = accuracy_score(MODE_TO_GT[mode], preds)
    model.train()
    return acc
