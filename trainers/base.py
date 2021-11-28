from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def __init__(self, config):
        raise NotImplementedError
    
    @abstractmethod
    def prepare(self, data):
        raise NotImplementedError
    
    @abstractmethod
    def fit(self):
        raise NotImplementedError
    