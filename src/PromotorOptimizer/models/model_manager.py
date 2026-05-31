from typing import Dict, List, Any

import numpy as np
import torch

from .base_model_manager import BaseModelManager
from ..utils.preprocessing import encode_batch


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

    # =====================================================
    # HELPERS
    # =====================================================

    def get_models(self):
        return self.models_dict

    def get_device(self):
        return self.device

    def get_model_names(self):
        return list(self.models_dict.keys())

    # =====================================================
    # TENSOR PREDICTION
    # =====================================================

    def predict_tensor(
        self,
        tensor: torch.Tensor
    ):

        outputs = {}

        tensor = tensor.to(self.device)

        for name, meta in self.models_dict.items():

            model = meta["model"]

            model.to(self.device)
            model.eval()

            with torch.no_grad():

                active, ratio = model(tensor)

            outputs[name] = {
                "active": active.detach().cpu(),
                "ratio": ratio.detach().cpu()
            }

        return outputs

    # =====================================================
    # RAW STRING PREDICTION
    # =====================================================

    def predict_sequences(
        self,
        sequences: List[str]
    ):

        X = encode_batch(sequences)

        X = torch.tensor(
            X,
            dtype=torch.float32,
            device=self.device
        )

        results = {
            seq: {}
            for seq in sequences
        }

        for model_name, meta in self.models_dict.items():

            model = meta["model"]

            model.to(self.device)
            model.eval()

            with torch.no_grad():

                active, ratio = model(X)

            ratio = (
                ratio
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )

            for idx, seq in enumerate(sequences):

                results[seq][model_name] = float(
                    ratio[idx]
                )

        return results

    # =====================================================
    # SINGLE SEQUENCE SCORE
    # =====================================================

    def score_sequence(
        self,
        sequence: str,
        penalty_std: float = 0.2
    ):

        scores = self.predict_sequences(
            [sequence]
        )[sequence]

        scores = np.array(
            list(scores.values())
        )

        mean = scores.mean()
        std = scores.std()

        return {
            "mean": float(mean),
            "std": float(std),
            "fitness": float(
                mean -
                penalty_std * std
            )
        }

    # =====================================================
    # MULTI SEQUENCE SCORE
    # =====================================================

    def score_sequences(
        self,
        sequences: List[str],
        penalty_std: float = 0.2
    ):

        raw = self.predict_sequences(
            sequences
        )

        output = {}

        for seq, model_scores in raw.items():

            scores = np.array(
                list(model_scores.values())
            )

            mean = scores.mean()
            std = scores.std()

            output[seq] = {
                "mean": float(mean),
                "std": float(std),
                "fitness": float(
                    mean -
                    penalty_std * std
                )
            }

        return output

    # =====================================================
    # ENSEMBLE STATS FROM TENSOR
    # =====================================================

    def ensemble_predict(
        self,
        tensor: torch.Tensor
    ):

        outputs = self.predict_tensor(
            tensor
        )

        scores = []

        for result in outputs.values():

            scores.append(
                result["ratio"]
                .mean()
                .item()
            )

        scores = np.array(scores)

        return {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max())
        }