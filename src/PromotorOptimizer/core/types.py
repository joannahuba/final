# core/types.py

from dataclasses import dataclass
from typing import Dict, Optional, List

@dataclass
class CandidateSequence:

    sequence: str

    original_sequence: str

    optimizer_source: str

    iteration: int

    mutation_count: int

    scores: Dict[str, float]

    ensemble_score: float

    is_valid: bool

    metadata: Optional[dict] = None

    
@dataclass
class InterpretationResult:

    method_name: str

    importance_scores: any

    model_name: Optional[str]

    metadata: Optional[dict] = None


@dataclass
class MutationRecord:
    """
    Represents a single mutation event.
    """

    position: int
    original_base: str
    new_base: str


@dataclass
class OptimizationStep:
    """
    Stores a single optimization iteration snapshot.
    """

    iteration: int

    sequence: str

    mutations: List[MutationRecord]

    model_scores: Dict[str, float]

    ensemble_score: float

    accepted: bool

    optimizer_name: str

    metadata: Optional[dict] = None


@dataclass
class ExperimentResult:
    """
    Full experiment result object.

    Stores:
    - final candidates
    - optimization history
    - interpretation outputs
    - model conflicts
    """

    sample_name: str

    original_sequence: str

    task_mode: str

    best_sequence: str

    best_score: float

    optimization_history: List[OptimizationStep]

    interpretations: Optional[dict] = None

    metadata: Optional[dict] = None