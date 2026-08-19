# main.py
import argparse
import logging

from src.algorithms.analysis_comparative import ComparativeAnalysisV4
from src.config import DATASETS
from src.visualizations.comparative_visualisation import generate_report, generate_visualizations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_analysis_for_dataset(dataset_path: str) -> None:
    """Exécute l'analyse comparative complète pour un seul dataset."""
    analyzer = ComparativeAnalysisV4(dataset_path)
    logger.info(f"Tailles testées : {analyzer.data_sizes}")

    try:
        analyzer.run_full_analysis()
    except KeyboardInterrupt:
        logger.warning("Analyse interrompue par l'utilisateur")
        return

    df = analyzer.create_dataframe()
    generate_visualizations(df, analyzer.output_dir, analyzer.data_sizes,
                             analyzer.support_ratios, analyzer.timeout_seconds)
    generate_report(df, analyzer.output_dir, analyzer.dataset_name, analyzer.test_structure,
                     analyzer.timeout_seconds, analyzer.grite_break_points, analyzer.grite_timeout_sizes)

    logger.info(f"Analyse terminée pour {analyzer.dataset_name}. Résultats dans : {analyzer.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyse comparative GGC vs GRITE")
    parser.add_argument(
        "--dataset", type=str, choices=list(DATASETS.keys()),
        help="Nom du dataset à analyser (voir src/config.py). Si omis, lance les 4.",
    )
    args = parser.parse_args()

    if args.dataset:
        run_analysis_for_dataset(str(DATASETS[args.dataset]))
    else:
        for name, path in DATASETS.items():
            logger.info(f"=== Dataset : {name} ===")
            run_analysis_for_dataset(str(path))


if __name__ == "__main__":
    main()