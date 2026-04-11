"""Prompt classifier: TF-IDF + Logistic Regression trained on answer-guided labels.

Usage:
    from src.classifier import PromptClassifier

    clf = PromptClassifier()
    clf.fit()                       # trains on full train set
    label = clf.predict("...")      # predicts a single prompt
    labels = clf.predict_batch(df)  # predicts a Series of prompts

The trained model is cached at models/classifier.pkl so it only
needs to be retrained when data changes.
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

from src.data_utils import load_train, classify_problem, PROBLEM_TYPES

MODEL_PATH = Path(__file__).parent.parent / "models" / "classifier.pkl"


class PromptClassifier:
    """TF-IDF character + word n-gram logistic regression classifier."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",   # character n-grams — great for structural patterns
                ngram_range=(2, 4),
                max_features=50_000,
                sublinear_tf=True,
            )),
            ("lr", LogisticRegression(
                max_iter=1000,
                C=5.0,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        self._fitted = False

    def fit(self, df: pd.DataFrame | None = None, verbose: bool = True) -> "PromptClassifier":
        """Train on answer-guided labels from the training set.

        Args:
            df: DataFrame with 'prompt' and 'problem_type' columns.
                Defaults to full training set with answer-guided labels.
            verbose: Print cross-val accuracy and classification report.
        """
        if df is None:
            df = load_train(use_answer_labels=True)

        X = df["prompt"]
        y = df["problem_type"]

        if verbose:
            scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="accuracy", n_jobs=-1)
            print(f"5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

        self.pipeline.fit(X, y)
        self._fitted = True

        if verbose:
            y_pred = self.pipeline.predict(X)
            print("\nTrain classification report:")
            print(classification_report(y, y_pred, target_names=sorted(y.unique())))

        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.pipeline, f)
        if verbose:
            print(f"\nModel saved to {MODEL_PATH}")

        return self

    def predict(self, prompt: str) -> str:
        """Predict problem type for a single prompt string."""
        self._ensure_fitted()
        return self.pipeline.predict([prompt])[0]

    def predict_batch(self, prompts: pd.Series) -> pd.Series:
        """Predict problem types for a Series of prompts."""
        self._ensure_fitted()
        return pd.Series(self.pipeline.predict(prompts), index=prompts.index)

    def predict_proba(self, prompt: str) -> dict[str, float]:
        """Return class probabilities for a single prompt."""
        self._ensure_fitted()
        classes = self.pipeline.classes_
        probs = self.pipeline.predict_proba([prompt])[0]
        return dict(zip(classes, probs))

    def _ensure_fitted(self):
        if not self._fitted:
            if MODEL_PATH.exists():
                self.load()
            else:
                raise RuntimeError("Classifier not fitted. Call clf.fit() first.")

    def load(self) -> "PromptClassifier":
        """Load a previously saved model from disk."""
        with open(MODEL_PATH, "rb") as f:
            self.pipeline = pickle.load(f)
        self._fitted = True
        return self

    @classmethod
    def load_or_fit(cls) -> "PromptClassifier":
        """Return a fitted classifier — loads from disk if available, else trains."""
        clf = cls()
        if MODEL_PATH.exists():
            clf.load()
        else:
            clf.fit()
        return clf


def evaluate_vs_keyword(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compare ML classifier against the keyword baseline on train set.

    Returns a DataFrame with per-type accuracy for both approaches.
    """
    if df is None:
        df = load_train(use_answer_labels=True)

    clf = PromptClassifier()
    clf.fit(df, verbose=False)

    df = df.copy()
    df["keyword_pred"] = df["prompt"].apply(classify_problem)
    df["ml_pred"] = clf.predict_batch(df["prompt"])
    df["gold"] = df["problem_type"]

    results = []
    for ptype in sorted(df["gold"].unique()):
        mask = df["gold"] == ptype
        n = mask.sum()
        kw_acc = (df.loc[mask, "keyword_pred"] == ptype).mean()
        ml_acc = (df.loc[mask, "ml_pred"] == ptype).mean()
        results.append({"type": ptype, "n": n, "keyword_acc": kw_acc, "ml_acc": ml_acc})

    result_df = pd.DataFrame(results)
    overall_kw = (df["keyword_pred"] == df["gold"]).mean()
    overall_ml = (df["ml_pred"] == df["gold"]).mean()
    result_df.loc[len(result_df)] = {
        "type": "OVERALL", "n": len(df),
        "keyword_acc": overall_kw, "ml_acc": overall_ml
    }
    return result_df
