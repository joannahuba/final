# models/registry.py

from typing import Dict, Any
import torch

from .loaders.deepstarr_loader import load_deepstarr
from .loaders.zero_loader import load_model_zero
from .datasets.dna_dataset import DNADataset


class ModelRegistry:

    @staticmethod
    def load(model_names: list[str]) -> Dict[str, Dict[str, Any]]:

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        registry = {}

        for name in model_names:

            # -------------------------
            # MODEL ZERO
            # -------------------------
            if name in ["model_zero", "zero_test_model"]:

                model = load_model_zero(
                    "data/checkpoints/zero_test_model.pth",
                    device
                )

                registry[name] = {
                    "model": model,
                    "dataset_class": DNADataset
                }

            # -------------------------
            # DEEPSTARR
            # -------------------------
            elif name in ["deepstarr", "scheduler_plateau"]:

                model = load_deepstarr(
                    "data/checkpoints/scheduler_plateau.pth",
                    device
                )

                registry[name] = {
                    "model": model,
                    "dataset_class": DNADataset
                }

            else:
                raise ValueError(f"Unknown model: {name}")

        return registry