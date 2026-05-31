from abc import ABC, abstractmethod


class BaseOptimizer(ABC):

    @abstractmethod
    def optimize(
        self,
        sequence,
        model_manager,
        interpretation,
        config
    ):
        pass