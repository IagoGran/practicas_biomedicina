import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
    from sdmetrics.single_table import DCROverfittingProtection, DisclosureProtectionEstimate
except ImportError as exc:
    QualityReport = None
    DiagnosticReport = None
    DCROverfittingProtection = None
    DisclosureProtectionEstimate = None
    SDMETRICS_IMPORT_ERROR = exc
else:
    SDMETRICS_IMPORT_ERROR = None


class SyntheticComparator:
    """Compare real and synthetic data across fidelity, utility and privacy."""

    def __init__(self, data_loader, synthetic_container, metadata_dict):
        if QualityReport is None or DiagnosticReport is None:
            raise ImportError(
                "SDMetrics is not installed. Run `pip install sdmetrics` before using SyntheticComparator."
            ) from SDMETRICS_IMPORT_ERROR

        self.data_loader = data_loader
        self.real_train = pd.concat(
            [
                data_loader.real_train_X.reset_index(drop=True),
                data_loader.real_train_y.reset_index(drop=True),
            ],
            axis=1,
        )
        self.real_test = pd.concat(
            [
                data_loader.real_test_X.reset_index(drop=True),
                data_loader.real_test_y.reset_index(drop=True),
            ],
            axis=1,
        )
        self.synthetic_train = pd.concat(
            [
                synthetic_container.X.reset_index(drop=True),
                synthetic_container.y.reset_index(drop=True),
            ],
            axis=1,
        )
        self.metadata = self._normalize_metadata_dict(metadata_dict)

    def _normalize_metadata_dict(self, metadata_dict):
        """Accept either SDMetrics single-table metadata or SDV V1 metadata."""
        if "columns" in metadata_dict:
            return metadata_dict

        if "tables" in metadata_dict:
            table_names = list(metadata_dict["tables"].keys())
            if len(table_names) != 1:
                raise ValueError(
                    "SyntheticComparator expects single-table metadata with exactly one table."
                )

            return metadata_dict["tables"][table_names[0]]

        raise ValueError("Unsupported metadata format. Expected a single-table metadata dictionary.")

    def quality_report(self):
        """Generate the full SDMetrics quality report for the synthetic training table."""
        report = QualityReport()
        report.generate(self.real_train, self.synthetic_train, self.metadata, verbose=False)

        return {
            "overall_score": report.get_score(),
            "properties": report.get_properties(),
            "column_shapes": report.get_details("Column Shapes"),
            "column_pair_trends": report.get_details("Column Pair Trends"),
        }

    def diagnostic_report(self):
        """Generate the SDMetrics diagnostic report for the synthetic training table."""
        report = DiagnosticReport()
        report.generate(self.real_train, self.synthetic_train, self.metadata, verbose=False)

        return {
            "overall_score": report.get_score(),
            "properties": report.get_properties(),
            "validity_details": report.get_details("Data Validity"),
            "structure_details": report.get_details("Data Structure"),
        }

    def privacy_dcr(self):
        """Measure privacy by checking whether synthetic rows are closer to train than holdout data."""
        return DCROverfittingProtection.compute_breakdown(
            real_training_data=self.real_train,
            synthetic_data=self.synthetic_train,
            real_validation_data=self.real_test,
            metadata=self.metadata,
        )

    def privacy_disclosure(self):
        """Estimate disclosure risk using a CAP-style attacker with a few known features."""
        known_columns = self.data_loader.feature_column_names[:3]
        target_column = self.data_loader.target_column_name

        return DisclosureProtectionEstimate.compute_breakdown(
            real_data=self.real_train,
            synthetic_data=self.synthetic_train,
            known_column_names=known_columns,
            sensitive_column_names=[target_column],
            continuous_column_names=known_columns,
            num_rows_subsample=500,
            num_iterations=20,
            verbose=False,
        )

    def utility_report(self):
        """Compare TR-TR and TS-TR utility using the same classifier on real test data."""
        tr_tr_metrics = self._train_and_evaluate(
            train_X=self.data_loader.real_train_X,
            train_y=self.data_loader.real_train_y,
            test_X=self.data_loader.real_test_X,
            test_y=self.data_loader.real_test_y,
        )
        ts_tr_metrics = self._train_and_evaluate(
            train_X=self.synthetic_train[self.data_loader.feature_column_names],
            train_y=self.synthetic_train[self.data_loader.target_column_name],
            test_X=self.data_loader.real_test_X,
            test_y=self.data_loader.real_test_y,
        )

        return {
            "tr_tr": tr_tr_metrics,
            "ts_tr": ts_tr_metrics,
            "delta": self._compute_metric_delta(tr_tr_metrics, ts_tr_metrics),
        }

    def quality_summary(self, quality_report):
        """Extract a compact summary from the full quality report."""
        properties = quality_report["properties"].set_index("Property")["Score"]
        return {
            "overall_score": float(quality_report["overall_score"]),
            "column_shapes_score": float(properties["Column Shapes"]),
            "column_pair_trends_score": float(properties["Column Pair Trends"]),
        }

    def diagnostic_summary(self, diagnostic_report):
        """Extract a compact summary from the full diagnostic report."""
        properties = diagnostic_report["properties"].set_index("Property")["Score"]
        return {
            "overall_score": float(diagnostic_report["overall_score"]),
            "data_validity_score": float(properties["Data Validity"]),
            "data_structure_score": float(properties["Data Structure"]),
        }

    def privacy_summary(self, dcr_report, disclosure_report):
        """Extract the main privacy signals from DCR and disclosure metrics."""
        return {
            "dcr_score": float(dcr_report["score"]),
            "closer_to_training": float(dcr_report["synthetic_data_percentages"]["closer_to_training"]),
            "closer_to_holdout": float(dcr_report["synthetic_data_percentages"]["closer_to_holdout"]),
            "disclosure_score": float(disclosure_report["score"]),
            "cap_protection": float(disclosure_report["cap_protection"]),
            "baseline_protection": float(disclosure_report["baseline_protection"]),
        }

    def _build_classifier(self):
        """Create the baseline classifier used for TR-TR and TS-TR comparisons."""
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

    def _train_and_evaluate(self, train_X, train_y, test_X, test_y):
        """Train the classifier on one dataset and evaluate it on real test data."""
        model = self._build_classifier()
        model.fit(train_X, train_y)

        predictions = model.predict(test_X)
        probabilities = model.predict_proba(test_X)[:, 1]

        return self._compute_classification_metrics(test_y, predictions, probabilities)

    def _compute_classification_metrics(self, y_true, y_pred, y_prob):
        """Compute the classification metrics used in the utility section."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        }

    def _compute_metric_delta(self, tr_tr_metrics, ts_tr_metrics):
        """Compute metric drops from training on real data to training on synthetic data."""
        return {
            metric_name: float(ts_tr_metrics[metric_name] - tr_tr_metrics[metric_name])
            for metric_name in tr_tr_metrics
        }
