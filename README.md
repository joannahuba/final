# Documentation

## Code Workflow

1. **Input Normalization:** Accepts promoter sequences as raw strings or nested configurations and standardizes them into a unified format.
2. **Trajectory Collection:** Passes sequences to optimizers and captures **all intermediate and final variants** (the entire mutation history) generated during the optimization process.
3. **Cascading Validation:** Filters the accumulated sequence pool using fast C-level Regex (`SequenceValidator`). Instantly flags variants that violate structural constraints (length boundaries, GC-content range, or homopolymer repeats).
4. **Virtual RAM Storage:** Streams all unique candidate sequences into an in-memory virtual TSV file (`tempfile`), bypassing disk I/O.
5. **Batch PyTorch Prediction:** Binds the virtual file to model-specific `Dataset` classes and feeds them via `DataLoader` to calculate forward-pass regression metrics (`predicted_rna_dna_ratio`) in parallel batches.
6. **Overfitting Assessment:** Computes a composite `min_model_score`

---

## Minimal Usage Template

```python
import torch
from wrapper import SequencePredictorModelWrapper

# 1. Prepare Target Models & Custom Datasets
loaded_model_1 = torch.load("path_to_model_alpha.pt")
loaded_model_2 = torch.load("path_to_model_beta.pt")

class NativeDNADataset:
    """Your project's Dataset class that tokenizes text into tensors."""
    pass

models_dict = {
    "model_alpha": {
        "model": loaded_model_1,
        "dataset_class": NativeDNADataset  # Pass the class reference, NOT an instance
    },
    "model_beta": {
        "model": loaded_model_2,
        "dataset_class": NativeDNADataset
    }
}

# 2. Setup Active Optimizers & Biological Constraints
sequence_optimizer = CustomEvolutionaryOptimizer()

validation_config = {
    "min_length": 20,
    "max_length": 60,
    "max_homopolymer_at": 4,
    "gc_percent_range": (0.4, 0.6)
}

# 3. Instantiate and Execute Pipeline
wrapper = SequencePredictorModelWrapper(
    models_dict=models_dict,
    optimizers_list=[sequence_optimizer],
    validation_config=validation_config
)

input_data = {
    "promoter_ref_01": "ATCGATCGATCGATCGATCGATCG"
}

results = wrapper.OptimizeSequence(input_data)
```

## Output Schema (proposal_table)
- The method returns a dictionary of lists. Converting a sample result directly to a pandas.DataFrame yields the following columns:
- sequence (str): The specific mutant variant from the optimization timeline.
- original_sequence (str): The wild-type/starting input sequence (baseline).
- optimizer_source (str): Class name of the optimizer that produced the variant.
- optimizer_internal_score (float | None): Metrics returned by the generator itself.
- score_[model_name] (float): Continuous evaluation ratio computed by that individual model.
- min_model_score (float): Core evaluation metric — minimum function over all model predictions.
- is_valid (bool): Physical/synthesis feasibility flag from the validator.