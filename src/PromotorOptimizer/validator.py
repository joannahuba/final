from typing import Callable, Dict, List, Optional, Tuple


class SequenceValidator:
    """
    Build a composite, callable sequence validator from a configuration.

    The class constructs small validator functions (homopolymer, GC-content,
    length) and composes them into a single callable object that accepts a
    list of sequences and returns per-sequence dictionaries with boolean
    results for each check and an overall `valid` flag.

    Configuration options (keys in `config`):
        - max_homopolymer_at (int|None): max consecutive A/T allowed
        - max_homopolymer_gc (int|None): max consecutive G/C allowed
        - gc_percent_range (tuple(float,float)|None): acceptable GC fraction (0..1)
        - max_length (int|None): maximum allowed sequence length
        - min_length (int|None): minimum allowed sequence length

    The instance itself is callable: `validator(list_of_seqs) -> list[dict]`.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.max_homopolymer_at = config.get("max_homopolymer_at")
        self.max_homopolymer_gc = config.get("max_homopolymer_gc")
        self.gc_percent_range = config.get("gc_percent_range")
        self.max_length = config.get("max_length")
        self.min_length = config.get("min_length")

        # Build list of check callables; each returns tuple(key, bool)
        self.checks: List[Callable[[str], Tuple[str, bool]]] = []

        if self.max_homopolymer_at is not None:
            self.checks.append(self._make_homopolymer_check("homopolymer_at", "AT", self.max_homopolymer_at))
        if self.max_homopolymer_gc is not None:
            self.checks.append(self._make_homopolymer_check("homopolymer_gc", "GC", self.max_homopolymer_gc))
        if self.gc_percent_range is not None:
            self.checks.append(self._make_gc_check("gc_content", self.gc_percent_range))
        if self.max_length is not None or self.min_length is not None:
            self.checks.append(self._make_length_check("length", self.min_length, self.max_length))

    # Factory methods return functions that evaluate single sequences and return (key, bool)
    def _make_homopolymer_check(self, key: str, bases: str, max_len: int) -> Callable[[str], Tuple[str, bool]]:
        def check(seq: str) -> Tuple[str, bool]:
            seq = seq.upper()
            current = 0
            for c in seq:
                if c in bases:
                    current += 1
                    if current > max_len:
                        return key, False
                else:
                    current = 0
            return key, True
        return check

    def _make_gc_check(self, key: str, gc_range: Tuple[float, float]) -> Callable[[str], Tuple[str, bool]]:
        def check(seq: str) -> Tuple[str, bool]:
            seq = seq.upper()
            if len(seq) == 0:
                return key, False
            gc = seq.count("G") + seq.count("C")
            frac = gc / len(seq)
            return key, (gc_range[0] <= frac <= gc_range[1])
        return check

    def _make_length_check(self, key: str, min_len: Optional[int], max_len: Optional[int]) -> Callable[[str], Tuple[str, bool]]:
        def check(seq: str) -> Tuple[str, bool]:
            l = len(seq)
            if min_len is not None and l < min_len:
                return key, False
            if max_len is not None and l > max_len:
                return key, False
            return key, True
        return check

    def __call__(self, sequences: List[str]) -> List[Dict[str, bool]]:
        """
        Validate a list of sequences.

        :param sequences: list of DNA sequence strings
        :return: list of dicts, each with keys for each check and `valid` boolean
        """
        results: List[Dict[str, bool]] = []
        # If no checks configured, return permissive results
        if not self.checks:
            for _ in sequences:
                results.append({
                    "valid": True
                })
            return results

        for seq in sequences:
            seq_result: Dict[str, bool] = {}
            for check in self.checks:
                key, ok = check(seq)
                seq_result[key] = ok
            # overall validity
            seq_result["valid"] = all(v for v in seq_result.values())
            results.append(seq_result)

        return results
