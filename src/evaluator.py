"""Local evaluation utilities for measuring solver accuracy."""

import pandas as pd
from src.data_utils import load_train


def exact_match(pred: str, gold: str) -> bool:
    return str(pred).strip().lower() == str(gold).strip().lower()


def evaluate(predictions: dict[str, str], df: pd.DataFrame | None = None) -> dict:
    """Evaluate predictions against ground truth.

    Args:
        predictions: {id: predicted_answer}
        df: DataFrame with 'id', 'answer', and 'problem_type' columns.
            Defaults to the full training set.

    Returns:
        dict with overall accuracy and per-type accuracy.
    """
    if df is None:
        df = load_train()

    df = df[df["id"].isin(predictions)].copy()
    df["pred"] = df["id"].map(predictions)
    df["correct"] = df.apply(lambda r: exact_match(r["pred"], r["answer"]), axis=1)

    overall = df["correct"].mean()
    per_type = df.groupby("problem_type")["correct"].mean().to_dict()

    print(f"Overall accuracy: {overall:.4f} ({df['correct'].sum()}/{len(df)})")
    for ptype, acc in sorted(per_type.items()):
        n = (df["problem_type"] == ptype).sum()
        print(f"  {ptype:<22} {acc:.4f}  (n={n})")

    return {"overall": overall, "per_type": per_type}
