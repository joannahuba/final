# optimizers/random_mutation_optimizer.py

from .optimizer_abstract import BaseSequenceOptimizer
from ..optimizers.mutation_generator import MutationGenerator


class RandomMutationOptimizer(BaseSequenceOptimizer):

    def __init__(self, n_candidates: int = 10):
        self.mut = MutationGenerator()
        self.n_candidates = n_candidates

    def optimize(self, sequence, evaluator, validator, interpreter=None, config=None):

        candidates = self.generate_candidates(sequence, config)
        scores = self.score_candidates(candidates, evaluator)
        selected = self.select_candidates(candidates, scores, config)

        return selected

    def generate_candidates(self, sequence, config=None):

        return [
            self.mut.random_mutation(sequence)
            for _ in range(self.n_candidates)
        ]

    def score_candidates(self, candidates, evaluator):

        scores = []

        for seq in candidates:
            # evaluator expects tensor OR string pipeline
            out = evaluator.predict_sequences([seq])
            score = list(out[seq].values())[0]
            scores.append(score)

        return scores

    def select_candidates(self, candidates, scores, config=None):

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        class Candidate:
            def __init__(self, sequence, score):
                self.sequence = sequence
                self.optimizer_source = "random_mutation"
                self.iteration = 0
                self.mutation_count = 1
                self.score = score

        return [Candidate(seq, s) for seq, s in ranked[:5]]