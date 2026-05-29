# pipeline/runner.py

import json
import pandas as pd

from ..models.model_manager import ModelManager
from ..models.registry import ModelRegistry
from ..optimizers.registry import OptimizerRegistry
from ..interpretation.registry import InterpreterRegistry

from ..core.wrapper import SequencePredictorModelWrapper
from .configs import PipelineConfig

from ..core.types import ExperimentResult
from .experiment_tracker import ExperimentTracker
from .exporters import ExperimentExporter


class PipelineRunner:
    """
    Single entrypoint for all experiments:
    - constrained recovery
    - unconstrained design
    """

    def __init__(self, config: PipelineConfig):

        self.config = config

        print("\n[INFO] Initializing pipeline")

        # 1. LOAD MODELS
        print(f"[INFO] Loading models: {config.models}")
        self.models_dict = ModelRegistry.load(config.models)
        self.model_manager = ModelManager(self.models_dict)

        # 2. LOAD OPTIMIZERS
        print(f"[INFO] Loading optimizers: {config.optimizers}")
        self.optimizers = OptimizerRegistry.load(config.optimizers)

        # 3. LOAD INTERPRETERS
        print(f"[INFO] Loading interpreters: {config.interpreters}")
        self.interpreters = InterpreterRegistry.load(config.interpreters)

        # 4. WRAPPER
        print("[INFO] Building wrapper")

        self.wrapper = SequencePredictorModelWrapper(
            model_manager=self.model_manager,
            optimizers_list=self.optimizers,
            validation_config=config.validation_config
        )

        print("[INFO] Pipeline initialized successfully")

    # -------------------------
    # MAIN ENTRYPOINT
    # -------------------------
    def run(self):

        print(f"\n[INFO] Reading input file: {self.config.input_path}")
        df = pd.read_csv(
            self.config.input_path,
            sep=r"\s+",
            header=None,
            names=["id", "sequence"]
        )

        print(f"[INFO] Loaded {len(df)} sequences")

        sequences = {
            row["id"]: row["sequence"]
            for _, row in df.iterrows()
        }

        results = {}

        print(f"[INFO] Task mode: {self.config.task_mode}")

        for name, seq in sequences.items():

            print(f"\n[INFO] Processing: {name}")
            print(f"[INFO] Sequence length: {len(seq)}")

            if self.config.task_mode == "constrained_recovery":
                output = self.wrapper.ReconstructSequence(
                    {name: seq},
                    mutation_n=self.config.mutation_budget
                )
            else:
                output = self.wrapper.OptimizeSequence(
                    {name: seq},
                    config={"iterations": self.config.iterations}
                )

            results[name] = output

        # -------------------------------------------------
        # FLATTEN RESULTS (IMPORTANT FIX)
        # -------------------------------------------------
        all_rows = []

        best_sequence = None
        best_score = float("-inf")

        for sample_name, table in results.items():
            for row in table[sample_name]:

                score = row["ensemble_score"]

                all_rows.append({
                    "sample_name": sample_name,
                    "sequence": row["sequence"],
                    "ensemble_score": score,
                    "scores": row["scores"],
                    "is_valid": row["is_valid"],
                })

                if score > best_score:
                    best_score = score
                    best_sequence = row["sequence"]

        # -------------------------------------------------
        # BUILD EXPERIMENT RESULT (FIXED)
        # -------------------------------------------------
        experiment = ExperimentResult(
            sample_name="batch_run",
            original_sequence=";".join(sequences.values()),
            task_mode=self.config.task_mode,
            best_sequence=best_sequence,
            best_score=float(best_score),
            optimization_history=all_rows,
            interpretations=None,
            metadata={
                "config": self.config.__dict__
            }
        )

        ExperimentExporter.export_json(
            experiment,
            self.config.output_path
        )

        return results