# pipeline/experiment_tracker.py

from typing import List, Dict
from ..core.types import OptimizationStep


class ExperimentTracker:

    def __init__(self):
        self.history: List[OptimizationStep] = []
        self.best_step_cache = None

    def log_step(self, step: OptimizationStep):
        self.history.append(step)

        if (
            self.best_step_cache is None
            or step.ensemble_score > self.best_step_cache.ensemble_score
        ):
            self.best_step_cache = step

    def best_step(self):
        return self.best_step_cache

    def export_curve(self):

        return [
            {
                "iteration": s.iteration,
                "ensemble_score": s.ensemble_score
            }
            for s in self.history
        ]

    def detect_model_conflicts(self, threshold: float = 0.5):

        conflicts = []

        for step in self.history:

            scores = list(step.model_scores.values())
            if len(scores) < 2:
                continue

            spread = max(scores) - min(scores)

            if spread >= threshold:
                conflicts.append({
                    "iteration": step.iteration,
                    "sequence": step.sequence,
                    "spread": spread,
                    "scores": step.model_scores
                })

        return conflicts