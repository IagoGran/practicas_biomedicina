from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None

SEED = 42

WDBC_COLUMNS = [
    "id",
    "diagnosis",
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
    "radius_error",
    "texture_error",
    "perimeter_error",
    "area_error",
    "smoothness_error",
    "compactness_error",
    "concavity_error",
    "concave_points_error",
    "symmetry_error",
    "fractal_dimension_error",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension",
]


class DataLoader:
    """Load, normalize and split the dataset from the UCI ML Repository."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = SEED,
        repo_id: int = 17,
        dataset_path: str | Path | None = None,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.repo_id = repo_id
        self.dataset_path = Path(dataset_path) if dataset_path else Path(__file__).resolve().parent.parent / "data" / "wdbc.data"
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
        X, y, metadata, variables = self._load_from_local_file()

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

    def _load_from_local_file(self):
        """Load the bundled WDBC dataset so the practice runs offline from the ZIP submission."""
        if not self.dataset_path.exists():
            return self._load_from_ucimlrepo()

        dataset = pd.read_csv(self.dataset_path, header=None, names=WDBC_COLUMNS)

        X = dataset.drop(columns=["id", "diagnosis"]).copy()
        y = self._normalize_target(dataset[["diagnosis"]])

        metadata = {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "source": "UCI Machine Learning Repository",
            "uci_id": self.repo_id,
            "local_path": str(self.dataset_path),
            "num_rows": len(dataset),
            "num_features": X.shape[1],
        }
        variables = pd.DataFrame(
            {
                "name": X.columns.tolist() + ["diagnosis"],
                "role": ["feature"] * X.shape[1] + ["target"],
                "type": ["continuous"] * X.shape[1] + ["categorical"],
            }
        )
        return X, y, metadata, variables

    def _load_from_ucimlrepo(self):
        """Fallback to the UCI repository only if the local copy is unavailable."""
        if fetch_ucirepo is None:
            raise ImportError(
                "ucimlrepo is not installed and the local dataset copy was not found."
            )

        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=self.repo_id)

        X = breast_cancer_wisconsin_diagnostic.data.features.copy()
        y = self._normalize_target(breast_cancer_wisconsin_diagnostic.data.targets)
        metadata = breast_cancer_wisconsin_diagnostic.metadata
        variables = breast_cancer_wisconsin_diagnostic.variables
        return X, y, metadata, variables

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
