"""Solver for bit manipulation puzzles.

Strategy: infer the sequence of bitwise operations (XOR, AND, OR, NOT,
shift, rotate) from the few-shot examples, then apply the same sequence
to the query input.
"""

from src.data_utils import split_examples, extract_query


def solve(prompt: str) -> str:
    """Return predicted answer for a bit manipulation prompt."""
    # TODO: implement operation inference from few-shot examples
    examples = split_examples(prompt)
    query = extract_query(prompt)
    raise NotImplementedError("bit_manipulation solver not yet implemented")
