from typing import Dict, List, Optional
import re


class SequenceValidator:
    """
    Build an optimized, composite, callable sequence validator from a configuration.

    The class processes a batch of sequences using a cascading short-circuit pipeline.
    It filters out invalid sequences on the fly, ensuring that expensive operations
    (like regex-based homopolymer searching) are only executed on sequences that passed
    cheaper, prior checks (like length and GC content).

    :param config: Configuration dictionary for the validator benchmarks.
    :type config: dict, optional


    Configuration options (keys in `config`):
        - max_homopolymer_at (int|None): max consecutive A/T allowed
        - max_homopolymer_gc (int|None): max consecutive G/C allowed
        - gc_percent_range (tuple(float,float)|None): acceptable GC fraction (0..1)
        - max_length (int|None): maximum allowed sequence length
        - min_length (int|None): minimum allowed sequence length

    Example:
        >>> config = {
        ...     "max_homopolymer_at": 5,
        ...     "max_homopolymer_gc": 4,
        ...     "gc_percent_range": (0.4, 0.6),
        ...     "min_length": 10,
        ...     "max_length": 100
        ... }
        >>> validator = SequenceValidator(config)
        >>> promoter_fragments = [
        ...     "ATCGATCGATCGATCGATCG",  # Valid sequence
        ...     "AAAAAAATCGATCGATCG",    # Invalid: Homopolymer AT (7x'A')
        ...     "CGCGCGCGCGCG",           # Invalid: Out of GC range
        ...     "ATCG",                   # Invalid: Too short
        ... ]
        >>> results = validator(promoter_fragments)
        >>> print(results)
        [True, False, False, False]
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the validator with optimized batch settings and pre-compiled regex patterns.

        :param config: Dictionary containing configuration thresholds
        """


        config = config or {}
        self.max_homopolymer_at = config.get("max_homopolymer_at")
        self.max_homopolymer_gc = config.get("max_homopolymer_gc")
        self.gc_percent_range = config.get("gc_percent_range")
        self.max_length = config.get("max_length")
        self.min_length = config.get("min_length")

        # Pre-compile regex patterns to execute sub-string searching directly in C.
        # Including both upper and lower case variants avoids expensive string allocation via .upper()
        self.at_pattern = (
            re.compile(f"[ATat]{{{self.max_homopolymer_at + 1}}}")
            if self.max_homopolymer_at is not None
            else None
        )
        self.gc_pattern = (
            re.compile(f"[GCgc]{{{self.max_homopolymer_gc + 1}}}")
            if self.max_homopolymer_gc is not None
            else None
        )

    def __call__(self, sequences: List[str]) -> List[bool]:
        """
        Validate a list of sequences sequentially by narrowing down active indices.

        Each step evaluates only the sequences that survived the previous check.
        Failed sequences are marked False immediately and skipped in subsequent steps.

        :param sequences: List of DNA sequence strings to validate
        :type sequences: list[str]
        :return: List of booleans matching the input length, where True means the sequence is valid
        :rtype: list[bool]
        """
        if not sequences:
            return []

        # Initialize the global status row for all input elements
        is_valid = [True] * len(sequences)
        
        # Track only the indices of sequences that are still valid
        active_indices = list(range(len(sequences)))

        # --- STEP 1: Length Check  ---
        if (self.min_length is not None or self.max_length is not None) and active_indices:
            next_active = []
            min_l, max_l = self.min_length, self.max_length
            for idx in active_indices:
                seq_len = len(sequences[idx])
                if (min_l is not None and seq_len < min_l) or (max_l is not None and seq_len > max_l):
                    is_valid[idx] = False
                else:
                    next_active.append(idx)
            active_indices = next_active  # Fold to survivors

        # --- STEP 2: GC Content Check ---
        if (self.gc_percent_range is not None) and active_indices:
            next_active = []
            gc_min, gc_max = self.gc_percent_range
            for idx in active_indices:
                seq = sequences[idx]
                seq_len = len(seq)
                
                # C-level native counting, avoids allocating new strings via .upper()
                gc_count = seq.count('G') + seq.count('C') + seq.count('g') + seq.count('c')
                frac = gc_count / seq_len if seq_len > 0 else 0.0
                
                if not (gc_min <= frac <= gc_max):
                    is_valid[idx] = False
                else:
                    next_active.append(idx)
            active_indices = next_active  # Fold to survivors

        # --- STEP 3: Homopolymer AT Check (Regex) ---
        if (self.at_pattern is not None) and active_indices:
            next_active = []
            search_at = self.at_pattern.search
            for idx in active_indices:
                if search_at(sequences[idx]):
                    is_valid[idx] = False
                else:
                    next_active.append(idx)
            active_indices = next_active  # Fold to survivors

        # --- STEP 4: Homopolymer GC Check  ---
        if (self.gc_pattern is not None) and active_indices:
            search_gc = self.gc_pattern.search
            for idx in active_indices:
                if search_gc(sequences[idx]):
                    is_valid[idx] = False

        return is_valid

# FOR DEBUGGING / EXAMPLE
if __name__ == "__main__":
    # Complete parameter pipeline setup
    config = {
        "max_homopolymer_at": 5,
        "max_homopolymer_gc": 4,
        "gc_percent_range": (0.4, 0.6),
        "min_length": 10,
        "max_length": 50
    }

    validator = SequenceValidator(config)

    # Mock dataset containing various edge-case failures
    promoter_fragments = [
        "ATCGATCGATCGATCGATCG",  # Valid (Length: 20, GC: 50%, No long homopolymers)
        "AAAAAAATCGATCGATCG",    # Invalid: Fails Homopolymer AT (7x 'A')
        "CGCGCGCGCGCG",           # Invalid: Fails GC Content (100% GC)
        "ATCG",                   # Invalid: Fails Length (Too short)
    ]

    # Execute batch pipeline directly
    results = validator(promoter_fragments)
    print("Validation mapping:", results)
    # Output: [True, False, False, False]