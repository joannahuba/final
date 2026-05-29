# interpretation/mutagenesis.py

import torch
from .base import BaseInterpreter


class InSilicoMutagenesis(BaseInterpreter):
    """
    For each position:
        - mutate
        - measure delta in output
    """

    def explain(self, model, sequence_tensor: torch.Tensor) -> torch.Tensor:

        model.eval()

        with torch.no_grad():

            _, base_ratio = model(sequence_tensor)
            base_score = base_ratio.mean(dim=-1)

            seq_len = sequence_tensor.shape[-1]
            importance = torch.zeros(seq_len)

            for i in range(seq_len):

                mutated = sequence_tensor.clone()

                # simple mutation: zero-out position (placeholder strategy)
                mutated[..., i] = 0.0

                _, ratio = model(mutated)
                score = ratio.mean(dim=-1)

                importance[i] = torch.abs(base_score - score)

        return importance.detach()