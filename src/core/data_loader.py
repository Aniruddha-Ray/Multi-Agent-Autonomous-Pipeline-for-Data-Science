"""Dataset acquisition for the Adaptive Multi-Agent Autonomous Data Science
Pipeline.

Loads `train.csv` and `test.csv` from `cfg.uploads_dir` by exact filename.
No synthetic fallback: if either file is missing, the caller must upload it
before the pipeline can run.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import Config


# def _find_uploaded_csv(uploads_dir: str) -> Optional[str]:
#     """Return the path of the first CSV found in ``uploads_dir``, if any."""
#     if not os.path.isdir(uploads_dir):
#         return None
#     csvs = sorted(glob.glob(os.path.join(uploads_dir, "*.csv")))
#     return csvs[0] if csvs else None


# def _try_colab_upload() -> Optional[str]:
#     """Attempt an interactive upload via Google Colab's ``files.upload()``."""
#     try:
#         from google.colab import files  # type: ignore
#     except ImportError:
#         return None
#     print("No CSV found in uploads_dir — please upload a CSV file.")
#     uploaded = files.upload()
#     for fname in uploaded.keys():
#         if fname.lower().endswith(".csv"):
#             return fname
#     return None


# def make_synthetic_dataset(random_state: int) -> pd.DataFrame:
#     """Generate a synthetic tabular classification dataset for fallback demo use."""
#     from sklearn.datasets import make_classification

#     X, y = make_classification(
#         n_samples=1200, n_features=10, n_informative=6, n_redundant=2,
#         n_clusters_per_class=2, weights=[0.85, 0.15], flip_y=0.02,
#         random_state=random_state,
#     )
#     rng = np.random.RandomState(random_state)
#     df = pd.DataFrame(X, columns=[f"num_feature_{i}" for i in range(X.shape[1])])
#     df["target"] = y

#     df["category_region"] = rng.choice(["North", "South", "East", "West"], size=len(df))
#     df["category_segment"] = rng.choice(["Retail", "Corporate", "SMB"], size=len(df))
#     df["category_id_highcard"] = rng.choice([f"ID_{i}" for i in range(60)], size=len(df))

#     for col in [c for c in df.columns if c.startswith("num_feature_")]:
#         mask = rng.rand(len(df)) < 0.07
#         df.loc[mask, col] = np.nan
#     mask = rng.rand(len(df)) < 0.04
#     df.loc[mask, "category_segment"] = np.nan

#     return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


# def load_dataset(cfg: Config) -> tuple[pd.DataFrame, str]:
#     """Resolve and load the dataset the pipeline will operate on."""
#     csv_path = _find_uploaded_csv(cfg.uploads_dir)
#     if csv_path:
#         return pd.read_csv(csv_path), f"user-uploaded CSV: {csv_path}"

#     csv_path = _try_colab_upload()
#     if csv_path:
#         return pd.read_csv(csv_path), f"Colab-uploaded CSV: {csv_path}"

#     print(f"No CSV found in '{cfg.uploads_dir}' and Colab upload unavailable/skipped — "
#           f"falling back to a synthetic dataset so the pipeline can run end-to-end.")
#     return make_synthetic_dataset(cfg.random_state), "synthetic fallback dataset (make_classification-based)"

def _resolve_one(uploads_dir: str, filename: str) -> str:
    """Return the path to `filename` inside `uploads_dir`, or raise if absent."""
    path = os.path.join(uploads_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"'{filename}' not found in '{uploads_dir}'. "
            f"Please place '{filename}' in that folder and rerun."
        )
    return path


def load_train_test(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Load `train.csv` and `test.csv` from `cfg.uploads_dir` by exact name."""
    train_path = _resolve_one(cfg.uploads_dir, "train.csv")
    test_path = _resolve_one(cfg.uploads_dir, "test.csv")
    return (
        pd.read_csv(train_path),
        pd.read_csv(test_path),
        f"user-uploaded train CSV: {train_path}",
        f"user-uploaded test CSV: {test_path}",
    )