# interpretation/integrated_gradients.py

import torch
from .base import BaseInterpreter


class IntegratedGradientsInterpreter(BaseInterpreter):

    def __init__(self, steps: int = 50):
        self.steps = steps

    def explain(self, model, sequence_tensor: torch.Tensor) -> torch.Tensor:
        """
        Integrated Gradients:
        baseline = zeros
        """

        baseline = torch.zeros_like(sequence_tensor)

        total_gradients = torch.zeros_like(sequence_tensor, dtype=torch.float32)

        for alpha in torch.linspace(0, 1, self.steps):

            x = (baseline + alpha * (sequence_tensor - baseline)).detach()
            x.requires_grad_(True)

            model.zero_grad()

            active, ratio = model(x)
            score = ratio.mean()

            score.backward()

            total_gradients += x.grad

        avg_grad = total_gradients / self.steps

        integrated_grads = (sequence_tensor - baseline) * avg_grad

        return integrated_grads.abs().mean(dim=-1).detach()