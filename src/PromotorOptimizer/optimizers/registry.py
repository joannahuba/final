# optimizers/registry.py

from .beam_search import BeamSearchOptimizer


class OptimizerRegistry:

    @staticmethod
    def load(names: list):

        registry = []

        for name in names:

            if name == "beam_search":
                registry.append(BeamSearchOptimizer())

            else:
                raise ValueError(f"Unknown optimizer {name}")

        return registry