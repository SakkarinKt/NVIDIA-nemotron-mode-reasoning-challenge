"""Shared data loading and problem classification utilities."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def classify_problem(prompt: str) -> str:
    """Classify a prompt into one of the known problem types."""
    p = prompt.lower()
    if any(k in p for k in ("bit", "xor", "binary", "shift", "rotate", "and ", "or ")):
        return "bit_manipulation"
    if any(k in p for k in ("encrypt", "decrypt", "cipher", "encoded", "decoded")):
        return "cipher"
    if any(k in p for k in ("roman", "numeral", "base ")):
        return "number_conversion"
    if any(k in p for k in ("meter", "kilogram", "celsius", "fahrenheit", "convert", "unit")):
        return "unit_conversion"
    return "symbol_transform"


def split_examples(prompt: str) -> list[tuple[str, str]]:
    """Parse the few-shot input->output pairs from a prompt.

    Returns a list of (input, output) string tuples.
    Handles lines like '01010001 -> 11011101' or 'hello -> world'.
    """
    examples = []
    for line in prompt.splitlines():
        if "->" in line:
            parts = line.split("->", 1)
            inp, out = parts[0].strip(), parts[1].strip()
            if inp and out:
                examples.append((inp, out))
    return examples


def extract_query(prompt: str) -> str:
    """Extract the final query (the input that needs an answer).

    The last line containing '->' with no right-hand side, or the last
    non-empty line if the prompt ends with a bare value.
    """
    lines = [l.strip() for l in prompt.splitlines() if l.strip()]
    for line in reversed(lines):
        if "->" in line:
            parts = line.split("->", 1)
            rhs = parts[1].strip()
            if not rhs:
                return parts[0].strip()
        else:
            # Bare query value (no arrow)
            return line
    return ""


def load_train() -> pd.DataFrame:
    """Load train.csv and add a problem_type column."""
    df = pd.read_csv(DATA_DIR / "train.csv")
    df["problem_type"] = df["prompt"].apply(classify_problem)
    return df


def load_test() -> pd.DataFrame:
    """Load test.csv and add a problem_type column."""
    df = pd.read_csv(DATA_DIR / "test.csv")
    df["problem_type"] = df["prompt"].apply(classify_problem)
    return df
