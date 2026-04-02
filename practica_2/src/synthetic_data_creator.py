import pandas as pd

try:
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer
except ImportError as exc:
    Metadata = None
    GaussianCopulaSynthesizer = None
    SDV_IMPORT_ERROR = exc
else:
    SDV_IMPORT_ERROR = None

from .data_classes import syntheticDataContainer, DataLoader


class SyntheticDataCreator:
    """
    A class to create synthetic data using the SDV library.
    It takes a DataLoader instance as input and generates a synthetic dataset based on the real training data.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        synthetic_data_container: syntheticDataContainer | None = None,
    ):
        if GaussianCopulaSynthesizer is None or Metadata is None:
            raise ImportError(
                "SDV is not installed. Run `pip install sdv` before using SyntheticDataCreator."
            ) from SDV_IMPORT_ERROR

        self.data_loader = data_loader
        self.real_train_table = self.data_loader.get_real_train_table()
        self.table_name = "breast_cancer_diagnostic"
        self.metadata = self._build_metadata()
        self.metadata_dict = self.metadata.to_dict()
        self.sdmetrics_metadata_dict = self._build_sdmetrics_metadata_dict()
        self.sdv_model = GaussianCopulaSynthesizer(metadata=self.metadata)

        if synthetic_data_container is not None:
            self.synthetic_data_container = synthetic_data_container
        else:
            self.synthetic_data_container = self.create_synthetic_data_container()

    def _build_metadata(self):
        """Detect metadata from the real training table and mark the target as categorical."""
        metadata = Metadata.detect_from_dataframe(
            data=self.real_train_table,
            table_name=self.table_name,
        )
        metadata.update_column(
            column_name=self.data_loader.target_column_name,
            sdtype="categorical",
        )
        metadata.validate()
        return metadata

    def _build_sdmetrics_metadata_dict(self) -> dict:
        """Convert SDV V1 metadata into the single-table dictionary expected by SDMetrics."""
        table_metadata = self.metadata_dict["tables"][self.table_name].copy()
        return table_metadata

    def create_synthetic_data_container(self) -> syntheticDataContainer:
        """
        Creates a synthetic data container by fitting the SDV model on the real training data and sampling from it.

        Returns:
            syntheticDataContainer: A container holding the synthetic features and target variable.
        """
        self.sdv_model.fit(self.real_train_table)

        synthetic_table = self.sdv_model.sample(num_rows=len(self.real_train_table))
        target_column = self.data_loader.target_column_name

        synthetic_X = synthetic_table.drop(columns=[target_column])
        synthetic_y = synthetic_table[target_column]

        if pd.api.types.is_numeric_dtype(self.data_loader.real_train_y):
            synthetic_y = synthetic_y.round().clip(
                lower=self.data_loader.real_train_y.min(),
                upper=self.data_loader.real_train_y.max(),
            ).astype(self.data_loader.real_train_y.dtype)

        synthetic_y.name = target_column
        return syntheticDataContainer(X=synthetic_X, y=synthetic_y)

    def get_synthetic_table(self) -> pd.DataFrame:
        """Return the synthetic train split as a single table."""
        return self.synthetic_data_container.to_table()
