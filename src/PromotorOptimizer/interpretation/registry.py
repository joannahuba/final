# interpretation/registry.py

from .saliency import SaliencyInterpreter
from .integrated_gradients import IntegratedGradientsInterpreter
from .mutagenesis import InSilicoMutagenesis


class InterpreterRegistry:

    @staticmethod
    def load(names: list):

        registry = []

        for name in names:

            if name == "saliency":
                registry.append(SaliencyInterpreter())

            elif name == "integrated_gradients":
                registry.append(IntegratedGradientsInterpreter())

            elif name == "mutagenesis":
                registry.append(InSilicoMutagenesis())

            else:
                raise ValueError(f"Unknown interpreter {name}")

        return registry