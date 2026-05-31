import random
import numpy as np


BASES = ["A", "C", "G", "T"]


class MutationGenerator:

    @staticmethod
    def mutate_position(
        sequence,
        position
    ):

        seq = list(sequence)

        current = seq[position]

        candidates = [
            b for b in BASES
            if b != current
        ]

        seq[position] = random.choice(
            candidates
        )

        return "".join(seq)

    @staticmethod
    def random_mutation(
        sequence,
        n_mutations=1
    ):

        seq = sequence

        positions = random.sample(
            range(len(sequence)),
            n_mutations
        )

        for pos in positions:
            seq = MutationGenerator.mutate_position(
                seq,
                pos
            )

        return seq

    @staticmethod
    def guided_mutation(
        sequence,
        importance_scores,
        n_mutations=1
    ):

        probs = (
            importance_scores.numpy()
            + 1e-8
        )

        probs /= probs.sum()

        positions = np.random.choice(
            len(sequence),
            size=n_mutations,
            replace=False,
            p=probs
        )

        seq = sequence

        for pos in positions:
            seq = MutationGenerator.mutate_position(
                seq,
                int(pos)
            )

        return seq

    @staticmethod
    def hybrid_mutation(
        sequence,
        importance_scores,
        n_mutations,
        lambda_weight=0.7
    ):

        guided_n = int(
            n_mutations * lambda_weight
        )

        random_n = (
            n_mutations - guided_n
        )

        seq = MutationGenerator.guided_mutation(
            sequence,
            importance_scores,
            guided_n
        )

        seq = MutationGenerator.random_mutation(
            seq,
            random_n
        )

        return seq