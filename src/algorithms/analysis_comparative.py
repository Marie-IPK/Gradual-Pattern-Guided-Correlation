"""
Analyse comparative des performances de GGC et GRITE sur plusieurs
configurations (taille de dataset, seuil de support, seuil de corrélation).

Ce module gère uniquement l'EXÉCUTION et la COLLECTE des résultats.
La génération de graphiques et de rapports est déléguée à
src/visualisation/comparative_visualisation.py.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.algorithms.ggc import CorrelationBasedGradualExtractor
from src.algorithms.grite import GriteExtractor
from src.config import (
    ANALYSIS_MAX_COLUMNS,
    ANALYSIS_TIMEOUT_SECONDS,
    CORRELATION_THRESHOLDS,
    CORRELATION_TYPE_FIXED,
    DATA_SIZES,
    SUPPORT_RATIOS,
    RESULTS_DIR,
)
from src.data.load_data import load_dataset
from src.data.processing import select_numeric_column
from src.utils import PerformanceMonitor, TimeoutException, get_timeout_snapshot, time_limit

logger = logging.getLogger(__name__)

# Noms des variables locales utilisées à l'intérieur de
# GriteExtractor.extract_gradual_patterns pour accumuler les motifs :
#   - frequent_pattern : motifs validés et fusionnés pour toutes les
#     tailles d'itemset déjà traitées
#   - new_patterns : motifs validés pour la taille en cours au moment du timeout
_KNOWN_GRITE_PATTERN_VARS = ("frequent_pattern", "new_patterns")


class ComparativeAnalysisV4:
    """Compare GGC et GRITE sur une grille de configurations (taille x support x corrélation)."""

    def __init__(self, dataset_path, output_dir="GGC_GRITE_Analysis"):
        self.dataset_path = dataset_path
        self.dataset_name = Path(dataset_path).stem
        self.output_dir = str(RESULTS_DIR / f"{output_dir}_{self.dataset_name}")
        self.results = []
        self.timeout_seconds = ANALYSIS_TIMEOUT_SECONDS

        self.grite_cache = {}
        self.grite_break_points = {}
        self.grite_timeout_sizes = set()

        self.data_sizes = DATA_SIZES
        self.support_ratios = SUPPORT_RATIOS
        self.correlation_thresholds = CORRELATION_THRESHOLDS
        self.correlation_type = CORRELATION_TYPE_FIXED
        self.max_columns = ANALYSIS_MAX_COLUMNS

        Path(self.output_dir).mkdir(exist_ok=True)
        self._generate_test_configuration_structure()

        logger.info(f"Dataset : {dataset_path}")
        logger.info(f"Dossier de sortie : {self.output_dir}")
        logger.info(f"Paires (taille, support) : {len(self.test_structure)}")

    def _generate_test_configuration_structure(self) -> None:
        """Génère la liste des paires (data_size, support_ratio) à tester."""
        self.test_structure = [
            {"data_size": size, "support_ratio": ratio}
            for size in self.data_sizes
            for ratio in self.support_ratios
        ]

    # ------------------------------------------------------------------
    # Chargement des données
    # ------------------------------------------------------------------

    def load_and_prepare_data(self, n_rows: int):
        """
        Charge le dataset et l'échantillonne pour une configuration de test.

        Réutilise load_dataset() et select_numeric_column() du pipeline de
        prétraitement standard ; le plafonnement du nombre de lignes/colonnes
        est spécifique à cette analyse (pas du prétraitement générique).
        """
        try:
            data = load_dataset(self.dataset_path)
            data = select_numeric_column(data)
            data = data.dropna().head(n_rows)
            data = data.iloc[:, : self.max_columns]
            return data
        except Exception as e:
            logger.error(f"Erreur de chargement des données : {e}")
            return None

    def calculate_support_threshold(self, data_size: int, support_ratio: float) -> int:
        """Calcule le support minimum absolu à partir d'un ratio et de la taille des données."""
        return max(2, int(data_size * support_ratio))

    # ------------------------------------------------------------------
    # Statistiques de motifs
    # ------------------------------------------------------------------

    def extract_pattern_statistics(self, patterns) -> dict:
        """Calcule des statistiques descriptives sur un ensemble de motifs extraits."""
        if not patterns:
            return {"total_patterns": 0, "max_itemset_size": 0, "k_itemsets": {}, "avg_support": 0}

        if not hasattr(patterns, "items"):
            # Récupération partielle après timeout : peut être une liste
            # plutôt qu'un dict — on ne peut alors donner qu'un compte.
            return {"total_patterns": len(patterns), "max_itemset_size": 0, "k_itemsets": {}, "avg_support": 0}

        patterns_by_size = defaultdict(int)
        supports = []
        k_itemsets = defaultdict(int)

        for pattern_key, pattern_data in patterns.items():
            if isinstance(pattern_data, tuple) and len(pattern_data) >= 2:
                size = len(pattern_key.split(",")) if isinstance(pattern_key, str) and "," in pattern_key else 1
                patterns_by_size[size] += 1
                k_itemsets[f"{size}-itemset"] += 1
                supports.append(pattern_data[1])

        return {
            "total_patterns": len(patterns),
            "max_itemset_size": max(patterns_by_size.keys()) if patterns_by_size else 0,
            "k_itemsets": dict(k_itemsets),
            "avg_support": np.mean(supports) if supports else 0,
        }

    # ------------------------------------------------------------------
    # Exécution des algorithmes
    # ------------------------------------------------------------------

    def run_ggc_algorithm(self, data, minsup: int, correlation_threshold: float) -> dict:
        """Exécute GGC pour une configuration donnée."""
        monitor = PerformanceMonitor()
        monitor.start()

        try:
            extractor = CorrelationBasedGradualExtractor(
                correlation_type=self.correlation_type,
                correlation_threshold=correlation_threshold,
            )
            patterns = extractor.extract_gradual_patterns(data, minsup, max_size=len(data.columns))

            metrics = monitor.get_metrics()
            stats = self.extract_pattern_statistics(patterns)

            return {
                "status": "success", "patterns": len(patterns),
                "statistics": stats, "performance": metrics, "error": None,
            }
        except Exception as e:
            metrics = monitor.get_metrics()
            return {
                "status": "error", "patterns": 0,
                "statistics": {"total_patterns": 0, "max_itemset_size": 0, "k_itemsets": {}, "avg_support": 0},
                "performance": metrics, "error": str(e),
            }

    def run_grite_algorithm(self, data, minsup: int) -> dict:
        """
        Exécute GRITE avec une limite de temps.

        En cas de timeout, tente de récupérer les motifs déjà accumulés
        (voir _KNOWN_GRITE_PATTERN_VARS) avant d'abandonner.
        """
        monitor = PerformanceMonitor()
        monitor.start()
        grite = GriteExtractor()

        try:
            with time_limit(self.timeout_seconds):
                patterns = grite.extract_gradual_patterns(data, minsup)

                metrics = monitor.get_metrics()
                stats = self.extract_pattern_statistics(patterns)

                return {
                    "status": "success", "patterns": len(patterns),
                    "statistics": stats, "performance": metrics,
                    "error": None, "timeout": False,
                }

        except TimeoutException:
            metrics = monitor.get_metrics()
            partial_patterns, recovery_method = self._recover_partial_grite_patterns(grite)
            stats = self.extract_pattern_statistics(partial_patterns)

            error_msg = f"Timeout après {self.timeout_seconds / 3600:.1f}h"
            error_msg += (
                f" -- résultats partiels récupérés depuis {recovery_method}"
                if recovery_method else " -- aucun résultat partiel récupérable"
            )

            return {
                "status": "timeout", "patterns": len(partial_patterns),
                "statistics": stats, "performance": metrics,
                "error": error_msg, "timeout": True,
            }

        except Exception as e:
            metrics = monitor.get_metrics()
            return {
                "status": "error", "patterns": 0,
                "statistics": {"total_patterns": 0, "max_itemset_size": 0, "k_itemsets": {}, "avg_support": 0},
                "performance": metrics, "error": str(e), "timeout": False,
            }

    def _recover_partial_grite_patterns(self, grite: GriteExtractor):
        """Tente de récupérer les motifs partiels d'un GRITE interrompu par timeout."""
        snapshot = get_timeout_snapshot()
        partial_patterns = {}
        recovery_method = None

        # 1) Variables connues de extract_gradual_patterns (fiable)
        if snapshot:
            known_found = {}
            for var_name, value, _depth in snapshot:
                if var_name in _KNOWN_GRITE_PATTERN_VARS and isinstance(value, dict):
                    known_found.setdefault(var_name, value)

            if known_found:
                merged = {}
                merged.update(known_found.get("frequent_pattern", {}))
                merged.update(known_found.get("new_patterns", {}))
                partial_patterns = merged
                recovery_method = f"variable(s) GRITE connue(s) : {', '.join(known_found.keys())}"

        # 2) Attributs d'instance courants, au cas où
        if not partial_patterns:
            for attr_name in ("patterns", "itemsets", "result", "results"):
                collected = getattr(grite, attr_name, None)
                if collected:
                    partial_patterns = collected
                    recovery_method = f"self.{attr_name}"
                    break

        # 3) Repli heuristique : plus grand dict/list trouvé sur la pile
        if not partial_patterns and snapshot:
            var_name, value, depth = max(snapshot, key=lambda item: len(item[1]))
            partial_patterns = value
            recovery_method = f"variable locale '{var_name}' (repli heuristique, profondeur {depth})"

        return partial_patterns, recovery_method

    def should_skip_grite(self, data_size: int, support_ratio: float):
        """Détermine si une configuration GRITE doit être sautée (uniquement si déjà en cache)."""
        config_key = (data_size, support_ratio)
        if config_key in self.grite_cache:
            return True, "cached"
        return False, None

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_pair_tests(self, data_size: int, support_ratio: float, pair_number: int, total_pairs: int) -> None:
        """Exécute toutes les variantes GGC puis GRITE pour une paire (taille, support)."""
        logger.info(f"[Paire {pair_number}/{total_pairs}] Taille : {data_size} | Support : {support_ratio * 100:.0f}%")

        data = self.load_and_prepare_data(data_size)
        if data is None:
            logger.error("Impossible de charger les données")
            return

        minsup = self.calculate_support_threshold(data_size, support_ratio)

        # --- Phase GGC : tous les seuils de corrélation ---
        for corr_threshold in self.correlation_thresholds:
            ggc_result = self.run_ggc_algorithm(data, minsup, corr_threshold)
            logger.info(
                f"GGC seuil={corr_threshold} : {ggc_result['status']} | "
                f"{ggc_result['performance']['execution_time']:.3f}s | {ggc_result['patterns']} motifs"
            )
            self.results.append({
                "data_size": data_size, "support_ratio": support_ratio,
                "correlation_threshold": corr_threshold, "minsup": minsup,
                "actual_data_size": len(data), "n_attributes": len(data.columns),
                "algorithm": "GGC", "ggc": ggc_result,
            })

        # --- Phase GRITE ---
        config_key = (data_size, support_ratio)
        should_skip, skip_reason = self.should_skip_grite(data_size, support_ratio)

        if should_skip:
            grite_result = self.grite_cache[config_key].copy()
            grite_result["status"] = "cached"
            logger.info(f"GRITE : ignoré ({skip_reason})")
        else:
            grite_result = self.run_grite_algorithm(data, minsup)
            logger.info(
                f"GRITE : {grite_result['status']} | "
                f"{grite_result['performance']['execution_time']:.3f}s | {grite_result['patterns']} motifs"
            )

            if grite_result["status"] == "timeout":
                self.grite_break_points.setdefault(data_size, support_ratio)
                self.grite_timeout_sizes.add(data_size)
            elif grite_result["status"] == "success":
                self.grite_cache[config_key] = grite_result.copy()

        self.results.append({
            "data_size": data_size, "support_ratio": support_ratio,
            "correlation_threshold": None, "minsup": minsup,
            "actual_data_size": len(data), "n_attributes": len(data.columns),
            "algorithm": "GRITE", "grite": grite_result,
        })

    def run_full_analysis(self) -> None:
        """Exécute l'analyse complète sur toutes les configurations de test."""
        total_pairs = len(self.test_structure)
        logger.info(f"Démarrage de l'analyse comparative — {total_pairs} paires à tester")

        for pair_number, cfg in enumerate(self.test_structure, 1):
            self.run_pair_tests(cfg["data_size"], cfg["support_ratio"], pair_number, total_pairs)

            if pair_number % 2 == 0:
                self.save_results()
                logger.info(f"[Sauvegarde intermédiaire] {pair_number}/{total_pairs}")

        self.save_results()
        logger.info(f"Analyse terminée. Résultats dans : {self.output_dir}")

    # ------------------------------------------------------------------
    # Sortie des résultats (données brutes, pas de présentation)
    # ------------------------------------------------------------------

    def save_results(self) -> None:
        """Sauvegarde les résultats bruts en JSON."""
        results_path = os.path.join(self.output_dir, "results.json")

        serializable_results = []
        for result in self.results:
            ser_result = {k: v for k, v in result.items() if k not in ["ggc", "grite"]}
            for algo_key in ["ggc", "grite"]:
                if algo_key in result:
                    algo_res = result[algo_key].copy()
                    algo_res["statistics"] = str(algo_res["statistics"])
                    algo_res["performance"] = str(algo_res["performance"])
                    ser_result[algo_key] = algo_res
            serializable_results.append(ser_result)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

    def create_dataframe(self) -> pd.DataFrame:
        """Construit un DataFrame détaillé à partir des résultats, pour l'analyse et la visualisation."""
        rows = []
        for result in self.results:
            base = {
                "data_size": result["data_size"],
                "support_ratio": result["support_ratio"],
                "support_pct": result["support_ratio"] * 100,
                "correlation_threshold": result.get("correlation_threshold"),
                "n_attributes": result["n_attributes"],
            }

            algo_name = result["algorithm"]
            algo_result = result.get(algo_name.lower())
            if algo_result:
                row = base.copy()
                row.update({
                    "algorithm": algo_name,
                    "status": algo_result["status"],
                    "execution_time": algo_result["performance"]["execution_time"],
                    "memory_used": algo_result["performance"]["memory_used"],
                    "total_patterns": algo_result["patterns"],
                    "max_itemset": algo_result["statistics"]["max_itemset_size"],
                })
                rows.append(row)

        return pd.DataFrame(rows)