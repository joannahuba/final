import torch
import pandas as pd
import tempfile

from torch.utils.data import DataLoader
from typing import Dict, List, Any

from .base_model_manager import BaseModelManager
from ..core.types import InterpretationResult


class ModelManager(BaseModelManager):

    def __init__(
        self,
        models_dict: Dict[str, Dict[str, Any]],
        batch_size: int = 64,
    ):
        self.models_dict = models_dict
        self.batch_size = batch_size

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    # -------------------------------------------------
    # 🔥 SINGLE SOURCE OF TRUTH FOR SHAPES
    # -------------------------------------------------
    def _fix_conv1d_shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensures model input is always (B, 4, L)
        Accepts:
        - (B, L, 4) -> permute
        - (B, 4, L) -> ok
        """
        if x.dim() == 3 and x.shape[-1] == 4:
            x = x.permute(0, 2, 1)
        return x

    # -------------------------------------------------
    # SINGLE TENSOR PREDICTION
    # -------------------------------------------------
    def predict(self, sequence_tensor: torch.Tensor):

        outputs = {}

        for name, meta in self.models_dict.items():

            model = meta["model"]
            model.to(self.device)
            model.eval()

            with torch.no_grad():

                x = sequence_tensor.to(self.device)
                x = self._fix_conv1d_shape(x)

                active, ratio = model(x)

            outputs[name] = {
                "active": active.detach().cpu(),
                "ratio": ratio.detach().cpu(),
            }

        return outputs

    # -------------------------------------------------
    # BATCH STRING PREDICTION
    # -------------------------------------------------
    def predict_sequences(self, sequences: List[str]):

        compiled_scores = {seq: {} for seq in sequences}

        with tempfile.NamedTemporaryFile(
            mode="w+",
            suffix=".tsv",
            delete=True
        ) as tmp_file:

            df = pd.DataFrame({
                "id": [f"seq_{i}" for i in range(len(sequences))],
                "sequence": sequences
            })

            df.to_csv(tmp_file.name, sep="\t", index=False)
            tmp_file.flush()

            for model_name, meta in self.models_dict.items():

                model = meta["model"]
                dataset_class = meta["dataset_class"]

                model.to(self.device)
                model.eval()

                dataset = dataset_class(
                    filepath=tmp_file.name,
                    is_test=True
                )

                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False
                )

                offset = 0

                with torch.no_grad():

                    for _, x in loader:

                        x = x.to(self.device)
                        x = self._fix_conv1d_shape(x)

                        active, ratio = model(x)

                        ratio = ratio.detach().cpu().numpy().flatten()

                        for i, score in enumerate(ratio):
                            seq = sequences[offset + i]
                            compiled_scores[seq][model_name] = float(score)

                        offset += len(ratio)

        return compiled_scores

    # -------------------------------------------------
    # ENSEMBLE
    # -------------------------------------------------
    def ensemble_predict(self, sequence_tensor: torch.Tensor):

        outputs = self.predict(sequence_tensor)

        scores = [
            out["ratio"].mean().item()
            for out in outputs.values()
        ]

        scores = torch.tensor(scores)

        return {
            "mean": float(scores.mean()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "std": float(scores.std() if len(scores) > 1 else 0.0),
        }

    # -------------------------------------------------
    # INTERPRETATION
    # -------------------------------------------------
    def explain(self, sequence_tensor, interpreter):

        results = {}

        for name, meta in self.models_dict.items():

            model = meta["model"]
            model.to(self.device)
            model.eval()

            importance = interpreter.explain(
                model,
                self._fix_conv1d_shape(sequence_tensor.to(self.device))
            )

            results[name] = InterpretationResult(
                method_name=interpreter.__class__.__name__,
                importance_scores=importance.detach().cpu(),
                model_name=name,
                metadata={}
            )

        return results

    # -------------------------------------------------
    # ENSEMBLE INTERPRETATION
    # -------------------------------------------------
    def explain_ensemble(self, sequence_tensor, interpreter):

        maps = []

        for meta in self.models_dict.values():

            model = meta["model"]
            model.to(self.device)
            model.eval()

            importance = interpreter.explain(
                model,
                self._fix_conv1d_shape(sequence_tensor.to(self.device))
            )

            maps.append(importance.unsqueeze(0))

        maps = torch.cat(maps, dim=0)

        return {
            "mean": maps.mean(dim=0),
            "std": maps.std(dim=0),
            "raw": maps
        }