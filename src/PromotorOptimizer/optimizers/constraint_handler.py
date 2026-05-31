import re


class ConstraintHandler:

    @staticmethod
    def gc_content(sequence):

        gc = (
            sequence.count("G")
            + sequence.count("C")
        )

        return gc / len(sequence)

    @staticmethod
    def valid_gc(
        sequence,
        lower=0.2,
        upper=0.8
    ):

        gc = ConstraintHandler.gc_content(
            sequence
        )

        return lower <= gc <= upper

    @staticmethod
    def has_long_repeat(
        sequence,
        max_repeat=6
    ):

        pattern = (
            r"(A{%d,}|C{%d,}|G{%d,}|T{%d,})"
            % (
                max_repeat,
                max_repeat,
                max_repeat,
                max_repeat
            )
        )

        return bool(
            re.search(
                pattern,
                sequence
            )
        )

    @staticmethod
    def is_valid(sequence):

        return (
            ConstraintHandler.valid_gc(
                sequence
            )
            and
            not ConstraintHandler.has_long_repeat(
                sequence
            )
        )