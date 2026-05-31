# interpretation/base.py

from abc import ABC, abstractmethod
from typing import Literal

from ..core.types import InterpretationResult


class BaseInterpreter(ABC):

    @abstractmethod
    def explain(
        self,
        model_manager,
        sequence: str,
        model_type: Literal["single", "ensemble"] = "ensemble"
    ) -> InterpretationResult:
        pass