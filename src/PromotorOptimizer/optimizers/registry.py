# optimizers/registry.py

from .random_mutation_optimizer import RandomMutationOptimizer
# from .genetic import GeneticOptimizer
# from .simulated_annealing import SimulatedAnnealingOptimizer


class OptimizerRegistry:

    @staticmethod
    def load(names: list):

        registry = []

        for name in names:

            if name == "mutation":
                registry.append(RandomMutationOptimizer())

            # elif name == "genetic":
            #     registry.append(GeneticOptimizer())

            # elif name == "simulated_annealing":
            #     registry.append(SimulatedAnnealingOptimizer())

            else:
                raise ValueError(f"Unknown optimizer {name}")

        return registry