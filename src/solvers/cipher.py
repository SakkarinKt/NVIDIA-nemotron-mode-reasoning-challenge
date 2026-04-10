"""Solver for cipher / encryption puzzles.

Strategy: detect the substitution mapping from the few-shot examples
(character-level alignment between plaintext and ciphertext), then
apply the inverse mapping to decode the query.
"""

from src.data_utils import split_examples, extract_query


def solve(prompt: str) -> str:
    """Return predicted answer for a cipher prompt."""
    # TODO: implement substitution map inference
    examples = split_examples(prompt)
    query = extract_query(prompt)
    raise NotImplementedError("cipher solver not yet implemented")
