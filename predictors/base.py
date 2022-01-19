from abc import ABC, abstractmethod

class BasePredictor(ABC):
    @abstractmethod
    def __init__(self, config):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, set_type):
        raise NotImplementedError
