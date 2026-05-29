# models/base_model_manager.py
from abc import ABC, abstractmethod
import torch


class BaseModelManager(ABC):

    @abstractmethod
    def predict(self, sequence_tensor: torch.Tensor):
        pass

    @abstractmethod
    def predict_sequences(self, sequences: list[str]):
        pass