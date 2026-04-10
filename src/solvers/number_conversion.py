"""Solver for number system conversion puzzles (e.g., decimal → Roman numerals).

Strategy: detect the conversion type from examples, then apply the
appropriate conversion function to the query.
"""

from src.data_utils import split_examples, extract_query


def solve(prompt: str) -> str:
    """Return predicted answer for a number conversion prompt."""
    # TODO: detect conversion direction and apply
    examples = split_examples(prompt)
    query = extract_query(prompt)
    raise NotImplementedError("number_conversion solver not yet implemented")
