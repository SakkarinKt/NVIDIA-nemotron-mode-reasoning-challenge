"""Solver for unit conversion puzzles.

Strategy: compute the conversion ratio from the few-shot examples,
then apply it to the query value.
"""

from src.data_utils import split_examples, extract_query


def solve(prompt: str) -> str:
    """Return predicted answer for a unit conversion prompt."""
    # TODO: infer ratio and apply to query
    examples = split_examples(prompt)
    query = extract_query(prompt)
    raise NotImplementedError("unit_conversion solver not yet implemented")
