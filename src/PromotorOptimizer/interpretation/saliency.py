# interpretation/saliency.py

import torch
from .base import BaseInterpreter


class SaliencyInterpreter(BaseInterpreter):
    """
    Gradient-based saliency:
    importance = d score / d input

    - signed_saliency: shows direction of influence
        (positive = increases prediction, negative = decreases prediction)
    - magnitude_saliency: absolute influence strength
    """

    def explain(self, model, sequence_tensor: torch.Tensor):

        # Enable gradients on input
        x = sequence_tensor.clone().detach().requires_grad_(True)

        model.zero_grad()

        active, ratio = model(x)

        # Scalar objective for backprop
        score = ratio.mean()

        score.backward()

        # Gradient w.r.t input
        grad = x.grad  # same shape as input

        # Collapse embedding/features dimension if present
        if grad.dim() > 1:
            signed_saliency = grad.mean(dim=-1)
        else:
            signed_saliency = grad

        # Keep sign (interpretability)
        signed_saliency = signed_saliency.detach()

        # Optional: magnitude-only importance
        magnitude_saliency = signed_saliency.abs()

        return {
            "signed_saliency": signed_saliency,
            "magnitude_saliency": magnitude_saliency
        }