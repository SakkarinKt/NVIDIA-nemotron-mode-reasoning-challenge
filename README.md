# NVIDIA Nemotron Model Reasoning Challenge

**Kaggle competition:** https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

Given a few-shot reasoning puzzle, predict the correct answer. Puzzles span 5 problem types:

| Type | Notebook |
|---|---|
| Bit Manipulation | `notebooks/01_bit_manipulation.ipynb` |
| Cipher / Encryption | `notebooks/02_cipher.ipynb` |
| Number Conversion | `notebooks/03_number_conversion.ipynb` |
| Unit Conversion | `notebooks/04_unit_conversion.ipynb` |
| Symbol Transform | `notebooks/05_symbol_transform.ipynb` |

## Setup

```bash
git clone <repo-url>
cd NVIDIA-nemotron-mode-reasoning-challenge
pip install -r requirements.txt
nbstripout --install   # strips notebook outputs to prevent merge conflicts
```

Download `train.csv` and `test.csv` from Kaggle and place them in `data/raw/`.

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, submission-ready code only |
| `develop` | Shared integration — all PRs merge here first |
| `feat/<type>/<desc>` | Feature work per problem type |

```
feat/* → PR → develop → (before submission) → main
```

## Running Tests

```bash
pytest tests/
```

## Generating a Submission

```bash
python src/generate_submission.py
# Output: submissions/submission_YYYY-MM-DD.csv — upload this to Kaggle
```
