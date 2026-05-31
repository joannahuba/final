import heapq

from .base_optimizer import BaseOptimizer
from .mutation_generator import MutationGenerator
from .constraint_handler import ConstraintHandler


class BeamSearchOptimizer(
    BaseOptimizer
):

    def __init__(
        self,
        beam_width=20,
        candidates_per_parent=10,
        iterations=25
    ):

        self.beam_width = beam_width
        self.candidates_per_parent = (
            candidates_per_parent
        )
        self.iterations = iterations

    def _score(
        self,
        model_manager,
        sequence
    ):

        result = (
            model_manager.predict_sequences(
                [sequence]
            )
        )

        scores = list(
            result[sequence].values()
        )

        mean_score = (
            sum(scores)
            / len(scores)
        )

        return mean_score

    def optimize(
        self,
        sequence,
        model_manager,
        interpretation,
        config
    ):

        beam = [sequence]

        trajectory = []

        best_score = self._score(
            model_manager,
            sequence
        )

        best_seq = sequence

        for iteration in range(
            self.iterations
        ):

            candidates = []

            for parent in beam:

                for _ in range(
                    self.candidates_per_parent
                ):

                    child = (
                        MutationGenerator.hybrid_mutation(
                            parent,
                            interpretation.importance_scores,
                            n_mutations=1,
                            lambda_weight=0.8
                        )
                    )

                    if not ConstraintHandler.is_valid(
                        child
                    ):
                        continue

                    score = self._score(
                        model_manager,
                        child
                    )

                    candidates.append(
                        (score, child)
                    )

            if not candidates:
                break

            beam = [
                seq
                for _, seq in heapq.nlargest(
                    self.beam_width,
                    candidates
                )
            ]

            current_best_score = max(
                candidates,
                key=lambda x: x[0]
            )[0]

            current_best_seq = max(
                candidates,
                key=lambda x: x[0]
            )[1]

            if (
                current_best_score
                > best_score
            ):
                best_score = (
                    current_best_score
                )
                best_seq = (
                    current_best_seq
                )

            trajectory.append(
                {
                    "iteration": iteration,
                    "sequence": best_seq,
                    "score": best_score,
                    "valid": True
                }
            )

        return {
            "best_sequence": best_seq,
            "best_score": best_score,
            "trajectory": trajectory
        }