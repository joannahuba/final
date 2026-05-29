# optimizers/optimizer_abstract.py

from abc import ABC, abstractmethod
from typing import List

class BaseSequenceOptimizer(ABC):

    def __call__(
        self,
        sequence: str,
        evaluator,
        validator,
        interpreter=None,
        config: dict | None = None,
    ):
        """
        Main public optimization API.
        """

        candidates = self.optimize(
            sequence=sequence,
            evaluator=evaluator,
            validator=validator,
            interpreter=interpreter,
            config=config,
        )

        return candidates

    @abstractmethod
    def optimize(
        self,
        sequence: str,
        evaluator,
        validator,
        interpreter=None,
        config: dict | None = None,
    ):
        """
        Main optimization loop.

        Returns:
            List[SequenceCandidate]
        """
        pass

    @abstractmethod
    def generate_candidates(
        self,
        sequence: str,
        config: dict | None = None,
    ):
        """
        Generate mutated candidate sequences.

        Returns:
            List[str]
        """
        pass

    @abstractmethod
    def score_candidates(
        self,
        candidates,
        evaluator,
    ):
        """
        Score candidate sequences.

        Returns:
            List[float]
        """
        pass

    @abstractmethod
    def select_candidates(
        self,
        candidates,
        scores,
        config: dict | None = None,
    ):
        """
        Select best candidates.

        Returns:
            List[SequenceCandidate]
        """
        pass