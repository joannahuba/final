# utils/encoding.py

import numpy as np

DNA_MAP = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
}


def clean_sequence(seq: str):
    seq = seq.upper()
    return "".join([b if b in DNA_MAP else "A" for b in seq])


def encode_one(seq: str):
    seq = clean_sequence(seq)
    return np.array([DNA_MAP[b] for b in seq], dtype=np.float32)


def encode_batch(seqs):
    return np.stack([encode_one(s) for s in seqs])