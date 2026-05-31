import torch

from .base import BaseInterpreter
from ..core.types import InterpretationResult
from ..utils.preprocessing import encode_one


class SaliencyInterpreter(BaseInterpreter):

    def explain(
        self,
        model_manager,
        sequence: str,
        model_type="ensemble"
    ):

        device = model_manager.get_device()
        models = model_manager.get_models()

        # (L, 4)
        
        x = torch.tensor(
            encode_one(sequence),
            dtype=torch.float32,
            device=device
        )
        # print("Encoded shape:", x.shape)

        # (1, L, 4)
        x = x.unsqueeze(0)

        saliency_maps = []
        model_scores = {}

        for name, meta in models.items():

            model = meta["model"]
            model.to(device)
            model.eval()

            x_req = x.clone().detach().requires_grad_(True)

            # print("Sequence length:", len(sequence))
            # print("Input tensor shape:", x_req.shape)
            active, ratio = model(x_req)

            score = ratio.mean()

            model_scores[name] = float(
                score.detach().cpu()
            )

            model.zero_grad()

            score.backward()

            grad = x_req.grad

            # (1, L, 4) -> (L,)
            saliency = grad.abs().sum(dim=2).squeeze(0)

            saliency_maps.append(saliency)

        saliency_maps = torch.stack(saliency_maps)

        if model_type == "ensemble":
            importance = saliency_maps.mean(dim=0)
        else:
            importance = saliency_maps[0]

        return InterpretationResult(
            method_name="Saliency",
            importance_scores=importance.detach().cpu(),
            sequence=sequence,
            model_scores=model_scores,
            metadata={}
        )