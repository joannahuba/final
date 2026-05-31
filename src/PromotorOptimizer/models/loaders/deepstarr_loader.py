# models/loaders/deepstarr_loader.py

import torch
from ..architectures.deepstarr import DeepSTARRLike


def load_deepstarr(checkpoint_path: str, device):

    model = DeepSTARRLike(
        seq_len=230,
        use_padding=True,
        dropout=0.4,
        use_logits=False
    )

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    # print("Loaded DeepSTARR")
    # print("fc1.in_features =", model.fc1.in_features)
    # print("_to_linear =", model._to_linear)
    return model