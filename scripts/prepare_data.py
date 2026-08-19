# scripts/prepare_data.py
"""
Prépare les 4 datasets bruts : charge, nettoie, sauvegarde dans data/processed/.
À exécuter une fois avant de lancer main.py (ou après toute modification des données brutes).
"""

import logging

from src.config import DATASETS
from src.data.load_data import load_dataset
from src.data.processing import pipeline_processing, save_processed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_all_datasets() -> None:
    """Charge, nettoie et sauvegarde chaque dataset défini dans config.DATASETS."""
    for name, raw_path in DATASETS.items():
        logger.info(f"Préparation du dataset : {name}")

        raw_data = load_dataset(raw_path)
        processed_data = pipeline_processing(raw_data)

        output_filename = f"{name}.csv"
        save_processed(processed_data, output_filename)


if __name__ == "__main__":
    prepare_all_datasets()