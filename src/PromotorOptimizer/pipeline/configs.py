# pipeline/configs.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal


TaskMode = Literal["constrained_recovery", "unconstrained_design"]


@dataclass
class PipelineConfig:
    input_path: str
    output_path: str

    task_mode: TaskMode

    mutation_budget: Optional[int] = None

    iterations: int = 50

    models: List[str] = field(default_factory=list)
    optimizers: List[str] = field(default_factory=list)
    interpreters: List[str] = field(default_factory=list)

    ensemble: bool = True

    validation_config: Optional[Dict] = None

    seed: int = 42