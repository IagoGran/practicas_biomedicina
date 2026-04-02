from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

SEED = 42


class DataLoader:
    """Load, normalize and split the dataset from the UCI ML Repository."""

    def __init__(self, test_size: float = 0.2, random_state: int = SEED, repo_id: int = 17):
        self.test_size = test_size
        self.random_state = random_state
        self.repo_id = repo_id
        (
            self.real_train_X,
            self.real_test_X,
            self.real_train_y,
            self.real_test_y,
            self.metadata,
            self.variables,
        ) = self.data_loader()
        self.target_column_name = self.real_train_y.name
        self.feature_column_names = list(self.real_train_X.columns)


    def data_loader(self):
        """
        Loads the dataset from the UCI Machine Learning Repository,
        splits it into training and testing sets,
        and returns the data along with metadata and variable information.
        """
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=self.repo_id)

        X = breast_cancer_wisconsin_diagnostic.data.features.copy()
        y = self._normalize_target(breast_cancer_wisconsin_diagnostic.data.targets)

        metadata = breast_cancer_wisconsin_diagnostic.metadata

        variables = breast_cancer_wisconsin_diagnostic.variables

        train_X, test_X, train_y, test_y = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        return (
            train_X.reset_index(drop=True),
            test_X.reset_index(drop=True),
            train_y.reset_index(drop=True),
            test_y.reset_index(drop=True),
            metadata,
            variables,
        )

    def _normalize_target(self, target_frame: DataFrame) -> Series:
        """Normalize the target column into a binary series usable by sklearn and SDV."""
        target_series = target_frame.iloc[:, 0].copy()
        target_series.name = "diagnosis"

        if target_series.dtype == "object":
            normalized = target_series.astype(str).str.strip().str.lower()
            mapping = {
                "b": 0,
                "benign": 0,
                "m": 1,
                "malignant": 1,
            }

            if normalized.isin(mapping.keys()).all():
                target_series = normalized.map(mapping)
            else:
                target_series = normalized

        return target_series.astype("int64") if pd.api.types.is_integer_dtype(target_series) or pd.api.types.is_bool_dtype(target_series) or target_series.dropna().isin([0, 1]).all() else target_series

    def get_real_train_table(self) -> DataFrame:
        """Return the real training data as a single table."""
        return pd.concat([self.real_train_X, self.real_train_y], axis=1)

    def get_real_test_table(self) -> DataFrame:
        """Return the real test data as a single table."""
        return pd.concat([self.real_test_X, self.real_test_y], axis=1)


@dataclass
class syntheticDataContainer:
    """
    A container class for synthetic data, which holds:
    - `X`: The feature data as a pandas DataFrame.
    - `y`: The target variable as a pandas Series.
    """

    X: DataFrame
    y: Series

    def to_table(self) -> DataFrame:
        """Return the synthetic data as a single table."""
        return pd.concat([self.X, self.y], axis=1)
