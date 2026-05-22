from typing import Dict, List, Callable, Optional, Tuple, Union

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
        models_dict: Dict,
        optimizers_list: List,
        kmer_sequences: Optional[List] = None,
        validation_config: Optional[Dict] = None,
    ):
        """
        Initialize the wrapper.

        :param models_dict: mapping of model name -> loaded model object
        :param optimizers_list: list of optimizer objects implementing ``__call__``
        :param kmer_sequences: optional k-mer list for optimizers
        :param validation_config: configuration passed to :class:`SequenceValidator`
        """
        if not models_dict:
            raise ValueError("models_dict cannot be empty")
        if not optimizers_list:
            raise ValueError("optimizers_list cannot be empty")

        self.models_dict = models_dict
        self.optimizers_list = optimizers_list
        self.kmer_sequences = kmer_sequences

        # Build validation callable (SequenceValidator is a callable class)
        self.validation_callable = self.SetValidationFunction(validation_config)

    def OptimizeSequence(
        #TODO pzepisać funkcje, to jest ogólny zarys, ale nie o to mi tutaj chodzi: 
        self,
        sequences: Dict[str, Union[str, Dict]],
        reconstruction_config: Optional[Dict] = None,
    ) -> Dict:
        """
        Optimize sequences using the registered optimizers.

        For the general optimize use-case the user supplies only sequence strings
        mapped by sample name. The wrapper will call each optimizer and let it
        decide how to modify the sequence (model-driven decisions).

        The method accepts two input formats for ``sequences``:
        - ``{'name': 'ATCG...'}`` (shorthand, treated as optimization)
        - ``{'name': {'sequence': 'ATCG...', 'method': 'optimization'}}``

        :param sequences: mapping of name -> sequence or name -> dict with keys
                          'sequence' and optional 'method'
        :param reconstruction_config: passed to optimizers when they need extra config
        :return: dictionary with optimization results per input name
        """
        if not sequences:
            raise ValueError("sequences cannot be empty")

        # normalize input: make each entry a dict with 'sequence' and 'method'
        normalized: Dict[str, Dict] = {}
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

        results: Dict = {}

        for sample_name, payload in normalized.items():
            seq = payload['sequence']
            method = payload['method'].lower()
            if method not in ('optimization', 'reconstruction'):
                raise ValueError("method must be 'optimization' or 'reconstruction'")

            proposals_collected = []
            for optimizer in self.optimizers_list:
                try:
                    proposals = optimizer(
                        sequence=seq,
                        method=method,
                        models_list=self.models_dict,
                        validation_function=self.validation_callable,
                        reconstruction_config=reconstruction_config,
                    )
                    proposals_collected.append({'optimizer': optimizer, 'proposals': proposals})
                except Exception as e:
                    print(f"Warning: optimizer {optimizer.__class__.__name__} failed: {e}")

            if not proposals_collected:
                results[sample_name] = {
                    'optimized_sequence': seq,
                    'original_sequence': seq,
                    'predictions': {},
                    'best_optimizer_score': None,
                    'is_valid': False,
                    'validation_results': {'valid': False},
                    'method': method,
                }
                continue

            best_seq, best_score = self._select_best_proposal(proposals_collected, seq)
            preds = self._evaluate_models(best_seq)
            validation = self.validation_callable([best_seq])[0]

            results[sample_name] = {
                'optimized_sequence': best_seq,
                'original_sequence': seq,
                'predictions': preds,
                'best_optimizer_score': best_score,
                'is_valid': validation.get('valid', False),
                'validation_results': validation,
                'method': method,
            }

        return results

    def ReconstructSequence(
        self,
        sequences: Dict[str, str],
        mutation_n: int,
        org_expression: Optional[float] = None,
        reconstruction_config: Optional[Dict] = None,
    ) -> Dict:
        """
        Reconstruct sequences given an exact number of mutations to introduce.

        :param sequences: mapping sample_name -> sequence (str)
        :param mutation_n: number of mutations to introduce during reconstruction
        :param org_expression: optional target expression value to aim for
        :param reconstruction_config: optimizer-specific reconstruction parameters
        :return: same format as :meth:`OptimizeSequence`
        """
        if not isinstance(mutation_n, int) or mutation_n < 0:
            raise ValueError("mutation_n must be a non-negative integer")

        # Merge mutation parameters into reconstruction_config
        rcfg = dict(reconstruction_config or {})
        rcfg.update({'mutation_n': mutation_n, 'org_expression': org_expression})

        # Convert to Normalize format expected by OptimizeSequence
        formatted = {name: seq for name, seq in sequences.items()}
        return self.OptimizeSequence(formatted, reconstruction_config=rcfg)

    def SetValidationFunction(self, validation_config: Optional[Dict]) -> Callable:
        """
        Instantiate and return a callable sequence validator.

        The validator is implemented in :class:`SequenceValidator` and is returned
        as a callable object. If ``validation_config`` is falsy, the validator
        will be permissive (accept all sequences).
        """
        return SequenceValidator(validation_config)

    # --- internal helpers ---
    # def _select_best_proposal(self, optimizer_proposals: List[Dict], original_sequence: str) -> Tuple[str, Optional[float]]:
    #     best_score = float('inf')
    #     best_sequence = original_sequence

    #     for entry in optimizer_proposals:
    #         proposals = entry.get('proposals')
    #         if isinstance(proposals, dict) and 'sequences' in proposals:
    #             for info in proposals['sequences']:
    #                 score = info.get('score', float('inf'))
    #                 candidate = info.get('sequence', original_sequence)
    #                 if score < best_score:
    #                     best_score = score
    #                     best_sequence = candidate

    #     if best_score == float('inf'):
    #         return best_sequence, None
    #     return best_sequence, best_score

    # def _evaluate_models(self, sequence: str) -> Dict[str, Tuple]:
    #     preds: Dict[str, Tuple] = {}
    #     for name, model in self.models_dict.items():
    #         try:
    #             if hasattr(model, 'evaluate'):
    #                 preds[name] = model.evaluate(sequence)
    #             else:
    #                 preds[name] = (None, None)
    #         except Exception as e:
    #             print(f"Warning: model {name} evaluation failed: {e}")
    #             preds[name] = (None, None)
    #     return preds



