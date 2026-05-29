# models/loaders/zero_loader.py

import torch
from ..architectures.genomic_model_zero import GenomicModelZero


def load_model_zero(checkpoint_path: str, device):

    model = GenomicModelZero()

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model