import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class DNADataset(Dataset):
    """
    Generic DNA dataset for inference and training.

    Expected TSV format:
        id \t sequence
    """

    def __init__(self, filepath: str, is_test: bool = True, header: int = 0):
        self.data = pd.read_csv(filepath, sep="\t", header=header)
        self.is_test = is_test

        if "sequence" not in self.data.columns:
            raise ValueError("Dataset must contain 'sequence' column")

        self.mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # -------------------------
    # ONE HOT ENCODING
    # -------------------------
    def one_hot_encode(self, seq: str):
        seq = seq.upper()

        encoded = np.zeros((4, len(seq)), dtype=np.float32)

        for i, base in enumerate(seq):
            if base in self.mapping:
                encoded[self.mapping[base], i] = 1.0

        return encoded

    # -------------------------
    def __len__(self):
        return len(self.data)

    # -------------------------
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        sequence = self.one_hot_encode(row["sequence"])
        sequence = torch.tensor(sequence, dtype=torch.float32)

        # inference mode
        if self.is_test:
            seq_id = str(row["id"]) if "id" in self.data.columns else str(idx)
            return seq_id, sequence

        # training mode (optional)
        ratio = torch.tensor(row.get("rna_dna_ratio", 0.0), dtype=torch.float32)
        active = torch.tensor(row.get("is_active", 0.0), dtype=torch.float32)

        return sequence, ratio, active