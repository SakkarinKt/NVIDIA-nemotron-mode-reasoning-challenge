"""Shared data loading and problem classification utilities."""

from pathlib import Path
import re
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# All 6 problem types found in the dataset
PROBLEM_TYPES = [
    "bit_manipulation",
    "cipher",
    "symbol_transform",
    "unit_conversion",
    "physics",
    "numeric",
]


def label_by_answer(answer: str, prompt: str) -> str:
    """Assign a reliable ground-truth label using the answer pattern.

    This is the gold labeling function — used to generate training labels
    for the ML classifier. Do NOT use at inference time (no answer available).
    """
    a = str(answer).strip()
    p = prompt.lower()

    # Binary string → bit manipulation
    if re.match(r'^[01]{4,}$', a):
        return "bit_manipulation"

    # Pure float/int → numeric subtypes
    if re.match(r'^-?\d+(\.\d+)?$', a):
        if "unit conversion" in p:
            return "unit_conversion"
        if "gravitational" in p:
            return "physics"
        return "numeric"

    # Roman numeral → symbol_transform (number conversion subtype)
    if re.match(r'^[IVXLCDM]+$', a):
        return "symbol_transform"

    # Words / phrase → cipher
    if re.match(r'^[a-zA-Z][a-zA-Z\s\-]+$', a):
        return "cipher"

    # Everything else (symbols, mixed chars) → symbol_transform
    return "symbol_transform"


def classify_problem(prompt: str) -> str:
    """Classify a prompt by text rules (used at inference on test set).

    Falls back to 'symbol_transform' when no strong signal is found.
    For higher accuracy, use the trained ML classifier in src/classifier.py.
    """
    p = prompt.lower()
    if "bit manipulation" in p or re.search(r'\b(xor|shift|rotat|bitwise)\b', p):
        return "bit_manipulation"
    if "encryption" in p or "secret encryption" in p:
        return "cipher"
    if "gravitational" in p:
        return "physics"
    if "unit conversion" in p:
        return "unit_conversion"
    if "numeral system" in p or "numeral" in p:
        return "symbol_transform"
    if "transformation rules" in p and re.search(r'\d+[-+*/]\d+', p):
        return "numeric"
    return "symbol_transform"


def split_examples(prompt: str) -> list[tuple[str, str]]:
    """Parse the few-shot input->output pairs from a prompt.

    Returns a list of (input, output) string tuples.
    Only returns complete pairs (both sides of the arrow non-empty).
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

    Looks for the last '->' line with an empty right-hand side,
    or falls back to the last non-empty line.
    """
    lines = [l.strip() for l in prompt.splitlines() if l.strip()]
    for line in reversed(lines):
        if "->" in line:
            parts = line.split("->", 1)
            rhs = parts[1].strip()
            if not rhs:
                return parts[0].strip()
        else:
            return line
    return ""


def load_train(use_answer_labels: bool = True) -> pd.DataFrame:
    """Load train.csv and add a problem_type column.

    Args:
        use_answer_labels: If True (default), use answer-guided labeling
            which is more accurate. If False, use prompt-only classify_problem.
    """
    df = pd.read_csv(DATA_DIR / "train.csv")
    if use_answer_labels:
        df["problem_type"] = df.apply(
            lambda r: label_by_answer(r["answer"], r["prompt"]), axis=1
        )
    else:
        df["problem_type"] = df["prompt"].apply(classify_problem)
    return df


def load_test() -> pd.DataFrame:
    """Load test.csv and classify prompts (no answer available)."""
    df = pd.read_csv(DATA_DIR / "test.csv")
    df["problem_type"] = df["prompt"].apply(classify_problem)
    return df
