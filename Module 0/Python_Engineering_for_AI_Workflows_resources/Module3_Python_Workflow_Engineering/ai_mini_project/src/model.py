"""
model.py

Defines the model classes used by this project.

Demonstrates: classes, inheritance/polymorphism, and @dataclass.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@dataclass
class TrainingConfig:
    """A typed, self-documenting container for training settings."""

    n_estimators: int = 100
    random_state: int = 42
    test_size: float = 0.2


class BaseModel:
    """
    Common interface every model in this project should follow.

    Subclasses must implement train() and predict(). This lets the rest
    of the codebase (train.py) work with any model the same way, without
    caring which specific algorithm is used underneath.
    """

    def train(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, X: pd.DataFrame):
        raise NotImplementedError("Subclasses must implement predict()")


class RandomForestModel(BaseModel):
    """A random forest classifier that follows the shared BaseModel interface."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return self.model.score(X, y)
