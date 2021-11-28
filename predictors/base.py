from abc import ABC, abstractmethod

class BasePredictor(ABC):
    def __init__(self, config):
        raise NotImplementedError

    def prepare(self, data):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, set_type):
        raise NotImplementedError
