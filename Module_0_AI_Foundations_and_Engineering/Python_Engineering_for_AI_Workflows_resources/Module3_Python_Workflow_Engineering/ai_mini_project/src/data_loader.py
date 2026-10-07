"""
data_loader.py

Handles all reading and cleaning of data for this project.

Demonstrates: file handling, type hints, and exception handling.
"""

from pathlib import Path

import pandas as pd


class DataValidationError(Exception):
    """Raised when a loaded dataset fails a basic quality check."""


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame and drop any rows with missing values.

    Raises:
        FileNotFoundError: if the given path does not exist.
        DataValidationError: if the file loads but contains no rows.
    """
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find data file: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna()

    if df.empty:
        raise DataValidationError(
            f"'{csv_path}' loaded successfully but contains no usable rows "
            "after removing missing values."
        )

    return df


def split_features_target(
    df: pd.DataFrame, target_col: str = "label"
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into a features table (X) and a target column (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
