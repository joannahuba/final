# optimizers/mutation_generator.py

import random
from typing import List, Dict, Optional


class MutationGenerator:
    """
    Generates DNA sequence mutations using:
    - random exploration
    - interpretation-guided mutation
    - k-mer replacement (biological prior)
    """

    def __init__(self, alphabet: Optional[List[str]] = None, seed: int = 42):

        # DNA alphabet default
        self.alphabet = alphabet or ["A", "C", "G", "T"]
        self.rng = random.Random(seed)

    # -------------------------------------------------
    # 1. RANDOM MUTATION
    # -------------------------------------------------
    def random_mutation(self, sequence: str) -> str:
        """
        Randomly mutates ONE position in the sequence.
        """

        if not sequence:
            return sequence

        seq = list(sequence)

        idx = self.rng.randint(0, len(seq) - 1)
        original = seq[idx]

        choices = [b for b in self.alphabet if b != original]
        seq[idx] = self.rng.choice(choices)

        return "".join(seq)

    # -------------------------------------------------
    # 2. GUIDED MUTATION (INTERPRETATION-DRIVEN)
    # -------------------------------------------------
    def guided_mutation(
        self,
        sequence: str,
        importance_scores,
    ) -> str:
        """
        Uses saliency / integrated gradients:
        - mutate HIGH importance positions
        """

        if not sequence:
            return sequence

        seq = list(sequence)

        # convert tensor → list
        scores = importance_scores.detach().cpu().numpy()

        # pick top-k important positions
        k = max(1, len(seq) // 10)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        for idx in top_indices:

            original = seq[idx]
            choices = [b for b in self.alphabet if b != original]

            if choices:
                seq[idx] = self.rng.choice(choices)

        return "".join(seq)

    # -------------------------------------------------
    # 3. K-MER REPLACEMENT (BIOLOGICAL PRIOR)
    # -------------------------------------------------
    def kmer_replacement(
        self,
        sequence: str,
        kmer_database: Dict[str, str]
    ) -> str:
        """
        Replace subsequences using known good k-mers.

        Example:
        kmer_database = {
            "ATG": "ATC",
            "CGT": "GGT"
        }
        """

        if not sequence or not kmer_database:
            return sequence

        seq = sequence

        # simple greedy replacement
        for kmer, replacement in kmer_database.items():
            if kmer in seq:
                seq = seq.replace(kmer, replacement)

        return seq