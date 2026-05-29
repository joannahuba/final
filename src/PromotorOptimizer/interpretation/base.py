# interpretation/base.py

from abc import ABC, abstractmethod
import torch


class BaseInterpreter(ABC):
    """
    All interpreters must return:
        torch.Tensor [seq_len]
    """

    @abstractmethod
    def explain(self, model, sequence_tensor: torch.Tensor) -> torch.Tensor:
        pass