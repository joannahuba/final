from typing import Any, Dict, List, Optional, Union
import tempfile
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .validator import SequenceValidator


class SequencePredictorModelWrapper:
    """
    Wrapper for torch-based sequence prediction models and sequence optimizers.

    This class exposes two separate public operations:
    - :meth:`OptimizeSequence` for general optimization tasks (model-driven)
    - :meth:`ReconstructSequence` for targeted reconstruction with a fixed
      number of mutations and an optional target expression

    The validation machinery is externalized into :class:`SequenceValidator`.
    """

    def __init__(
        self,
        models_dict: Dict[str, Dict[str, Any]],
        optimizers_list: List[Any],
        kmer_sequences: Optional[List[str]] = None,
        validation_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the wrapper with model dictionaries and optimizer pipelines.

        :param models_dict: Mapping of model_name -> {'model': loaded_model, 'dataset_class': dataset_cls}
        :type models_dict: dict
        :param optimizers_list: List of optimizer objects implementing __call__
        :type optimizers_list: list
        :param kmer_sequences: Optional k-mer lookup table for active optimizers, defaults to None
        :type kmer_sequences: list, optional
        :param validation_config: Configuration thresholds passed to SequenceValidator, defaults to None
        :type validation_config: dict, optional
        """
        if not models_dict:
            raise ValueError("models_dict cannot be empty")
        if not optimizers_list:
            raise ValueError("optimizers_list cannot be empty")

        self.models_dict = models_dict
        self.optimizers_list = optimizers_list
        self.kmer_sequences = kmer_sequences

        # Build validation callable (SequenceValidator is a callable class)
        self._ValidationFunction = self.SetValidationFunction(validation_config)

    def OptimizeSequence(
        self,
        sequences: Dict[str, Union[str, Dict[str, Any]]],
        reconstruction_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Optimize sequences using the registered optimizers and score them across all models.

        The method accepts two input formats for ``sequences``:
        - ``{'name': 'ATCG...'}`` (shorthand, treated as optimization)
        - ``{'name': {'sequence': 'ATCG...', 'method': 'optimization'}}``

        :param sequences: Mapping of sample name to sequence string or detailed configuration payload.
        :type sequences: dict
        :param reconstruction_config: Contextual hyper-parameters passed to sequence optimizers, defaults to None
        :type reconstruction_config: dict, optional
        :return: Tabular matrix output sorted by model evaluations per sample reference.
        :rtype: dict
        """
        if not sequences:
            raise ValueError("sequences cannot be empty")

        #
        ## Input data normalization into a unified dictionary layout
        normalized: Dict[str, Dict[str, Any]] = {}
        for name, value in sequences.items():
            if isinstance(value, str):
                normalized[name] = {'sequence': value, 'method': 'optimization'}
            elif isinstance(value, dict):
                if 'sequence' not in value:
                    raise KeyError(f"Entry for {name} must contain 'sequence'")
                method = value.get('method', 'optimization')
                normalized[name] = {'sequence': value['sequence'], 'method': method}
            else:
                raise TypeError("Sequence value must be str or dict")

        results: Dict[str, List[Dict[str, Any]]] = {}

        for sample_name, payload in normalized.items():
            seq = payload['sequence']
            method = payload['method'].lower()
            if method not in ('optimization', 'reconstruction'):
                raise ValueError("method must be 'optimization' or 'reconstruction'")

            # #
            # ## Execute proposals collection loop across all registered optimizers
            # #
            proposals_collected = []
            for optimizer in self.optimizers_list:
                try:
                    proposals = optimizer(
                        sequence=seq,
                        method=method,
                        models_list=self.models_dict,
                        validation_function=self.ValidationFunction,
                        reconstruction_config=reconstruction_config,
                    )
                    proposals_collected.append({'optimizer': optimizer, 'proposals': proposals})
                except Exception as e:
                    print(f"Warning: optimizer {optimizer.__class__.__name__} failed: {e}")

            if not proposals_collected:
                # Fallback return payload if optimizers failed to execute
                results[sample_name] = [{
                    'sequence': seq,
                    'original_sequence': seq,
                    'optimizer_source': 'failed_pipeline',
                    'optimizer_internal_score': None,
                    'min_model_score': float('inf'),
                    'is_valid': False,
                    'method': method,
                }]
                continue

            
            ## Build global candidate sequences set and parse dynamic historical metrics
            candidate_sequences_set = set()
            optimizer_meta = {}  

            for entry in proposals_collected:
                opt_name = entry['optimizer'].__class__.__name__
                proposals = entry.get('proposals')
                
                ## Case A: Standard dictionary response containing a list of sequence blocks
                if isinstance(proposals, dict) and 'sequences' in proposals:
                    for info in proposals['sequences']:
                        candidate = info.get('sequence')
                        if candidate:
                            candidate_sequences_set.add(candidate)
                            if candidate not in optimizer_meta:
                                optimizer_meta[candidate] = {
                                    'optimizer_name': opt_name,
                                    'optimizer_score': info.get('score')
                                }
                ## Case B: Flat list array of raw string sequences or training trace dictionaries
                elif isinstance(proposals, list):
                    for info in proposals:
                        candidate = info if isinstance(info, str) else (isinstance(info, dict) and info.get('sequence'))
                        if candidate:
                            candidate_sequences_set.add(candidate)
                            if candidate not in optimizer_meta:
                                optimizer_meta[candidate] = {
                                    'optimizer_name': opt_name,
                                    'optimizer_score': info.get('score') if isinstance(info, dict) else None
                                }

            ## Ensure original baseline sequence is anchored inside the pool
            candidate_sequences_set.add(seq)
            if seq not in optimizer_meta:
                optimizer_meta[seq] = {'optimizer_name': 'baseline', 'optimizer_score': None}

            candidate_list = list(candidate_sequences_set)

            ## Perform vectorized validation passes across the distinct pool
            validation_outputs = self.ValidationFunction(candidate_list)
            validation_map = {cand: val for cand, val in zip(candidate_list, validation_outputs)}

            compiled_scores = {cand: {} for cand in candidate_list}
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) 

            ## Create virtual file storage session to feed data-loaders seamlessly
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.tsv', delete=True) as tmp_file:
                mock_df = pd.DataFrame({
                    'id': [f"cand_{i}" for i in range(len(candidate_list))],
                    'sequence': candidate_list
                })
                mock_df.to_csv(tmp_file.name, sep='\t', index=False)
                tmp_file.flush()

                ## Run evaluation loops model-by-model across in-memory batches
                for model_name, model_meta in self.models_dict.items():
                    model_obj = model_meta['model']
                    dataset_class = model_meta['dataset_class']

                    model_obj.to(device)
                    model_obj.eval()

                    try:
                        eval_dataset = dataset_class(filepath=tmp_file.name, is_test=True)
                        eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

                        with torch.no_grad():
                            batch_idx_offset = 0
                            for batch in eval_loader:
                                _, sequences_tensor = batch
                                sequences_tensor = sequences_tensor.to(device)

                                ## Execute forward pass mapping custom structural prediction heads
                                _, pred_ratios = model_obj(sequences_tensor)
                                ratios_numpy = pred_ratios.cpu().numpy().flatten()

                                for local_idx, score_value in enumerate(ratios_numpy):
                                    global_idx = batch_idx_offset + local_idx
                                    target_sequence = candidate_list[global_idx]
                                    compiled_scores[target_sequence][f'score_{model_name}'] = float(score_value)

                                batch_idx_offset += len(ratios_numpy)

                    except Exception as eval_ex:
                        print(f"Warning: Batch evaluation failed for model {model_name}: {eval_ex}")
                        for cand in candidate_list:
                            compiled_scores[cand][f'score_{model_name}'] = float('inf')

            ## Compile the structured data results matrix containing calculated minimums
            proposal_table: List[Dict[str, Any]] = []

            for candidate in candidate_list:
                model_metrics = compiled_scores[candidate]
                valid_scores = [v for v in model_metrics.values() if v != float('inf')]
                min_model_score = min(valid_scores) if valid_scores else float('inf')

                is_valid_bool = (
                    validation_map[candidate] 
                    if isinstance(validation_map[candidate], bool) 
                    else validation_map[candidate].get('valid', False)
                )

                row = {
                    'sequence': candidate,
                    'original_sequence': seq,
                    'optimizer_source': optimizer_meta[candidate]['optimizer_name'],
                    'optimizer_internal_score': optimizer_meta[candidate]['optimizer_score'],
                    **model_metrics,
                    'min_model_score': min_model_score,
                    'is_valid': is_valid_bool,
                    'method': method
                }
                proposal_table.append(row)

            results[sample_name] = proposal_table

        return results

    def ReconstructSequence(
        self,
        sequences: Dict[str, str],
        mutation_n: int,
        org_expression: Optional[float] = None,
        reconstruction_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Reconstruct sequences given an exact number of mutations to introduce.

        :param sequences: Mapping sample_name -> sequence string
        :type sequences: dict
        :param mutation_n: Number of target mutations allowed during alignment passes
        :type mutation_n: int
        :param org_expression: Optional target benchmark expression value, defaults to None
        :type org_expression: float, optional
        :param reconstruction_config: Secondary runtime config parameters, defaults to None
        :type reconstruction_config: dict, optional
        :return: Processed tabular scoring results matching the schema of OptimizeSequence
        :rtype: dict
        """
        if not isinstance(mutation_n, int) or mutation_n < 0:
            raise ValueError("mutation_n must be a non-negative integer")

        rcfg = dict(reconstruction_config or {})
        rcfg.update({'mutation_n': mutation_n, 'org_expression': org_expression})

        formatted = {name: seq for name, seq in sequences.items()}
        return self.OptimizeSequence(formatted, reconstruction_config=rcfg)

    def SetValidationFunction(self, validation_config: Optional[Dict[str, Any]]) -> SequenceValidator:
        """
        Instantiate and return a callable sequence validator engine instance.

        :param validation_config: Dictionary containing parameter thresholds
        :type validation_config: dict, optional
        :return: Initialized SequenceValidator object
        :rtype: SequenceValidator
        """
        return SequenceValidator(validation_config)
    
    def ValidationFunction(self, sequences: List[str]) -> List[bool]:
        """
        Execute continuous sequence validation checks.

        :param sequences: List of string sequences to filter
        :type sequences: list
        :return: Execution results list containing status mappings or booleans
        :rtype: list
        """
        return self._ValidationFunction(sequences)