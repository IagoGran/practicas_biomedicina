import warnings

from src.data_classes import DataLoader
from src.synthetic_comparator import SyntheticComparator
from src.synthetic_data_creator import SyntheticDataCreator


def configure_warnings():
    """Hide noisy third-party warnings that do not affect the assignment workflow."""
    warnings.filterwarnings(
        "ignore",
        message="We strongly recommend saving the metadata using 'save_to_json'.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Your real_validation_data contains .*",
        category=UserWarning,
    )


def print_section(title: str):
    """Print a readable section header for the command-line output."""
    print(f"\n{title}")
    print("-" * len(title))


def print_metric_block(metrics: dict):
    """Print a small dictionary of numeric metrics with fixed precision."""
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def print_utility_report(utility_report: dict):
    """Print the TR-TR and TS-TR utility results in a compact format."""
    print("TR-TR")
    print_metric_block(utility_report["tr_tr"])
    print("\nTS-TR")
    print_metric_block(utility_report["ts_tr"])
    print("\nDelta (TS-TR - TR-TR)")
    print_metric_block(utility_report["delta"])


def main():
    """Run the end-to-end synthetic data evaluation pipeline."""
    configure_warnings()
    data_loader = DataLoader()
    synthetic_data_creator = SyntheticDataCreator(data_loader)
    synthetic_data_container = synthetic_data_creator.synthetic_data_container
    comparator = SyntheticComparator(
        data_loader,
        synthetic_data_container,
        synthetic_data_creator.sdmetrics_metadata_dict,
    )

    quality_report = comparator.quality_report()
    diagnostic_report = comparator.diagnostic_report()
    privacy_dcr_report = comparator.privacy_dcr()
    privacy_disclosure_report = comparator.privacy_disclosure()
    utility_report = comparator.utility_report()

    print_section("Fidelity")
    print_metric_block(comparator.quality_summary(quality_report))

    print_section("Diagnostic")
    print_metric_block(comparator.diagnostic_summary(diagnostic_report))

    print_section("Privacy")
    print_metric_block(comparator.privacy_summary(privacy_dcr_report, privacy_disclosure_report))

    print_section("Utility")
    print_utility_report(utility_report)

if __name__ == "__main__":
    main()
