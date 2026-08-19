# -*- coding: utf-8 -*-

import logging
from collections import defaultdict, deque
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from src.utils import PerformanceMonitor
from src.algorithms.common import generate_binary_matrix_vectorized
logger = logging.getLogger(__name__)

SUPPORTED_CORRELATION_TYPES = ("pearson", "spearman", "kendall")


class CorrelationBasedGradualExtractor:
    """Extracteur de motifs graduels basé sur les corrélations entre attributs."""

    def __init__(self, correlation_type: str = "pearson", correlation_threshold: float = 0.5):
        if correlation_type not in SUPPORTED_CORRELATION_TYPES:
            raise ValueError(
                f"Type de corrélation non supporté : '{correlation_type}'. "
                f"Choix possibles : {SUPPORTED_CORRELATION_TYPES}"
            )

        self.cache = {}
        self.correlation_type = correlation_type
        self.correlation_threshold = abs(correlation_threshold)
        self.monitor = PerformanceMonitor()

        self.correlation_functions = {
            "pearson": pearsonr,
            "spearman": spearmanr,
            "kendall": kendalltau,
        }

    # ------------------------------------------------------------------
    # ÉTAPE 1 : corrélations
    # ------------------------------------------------------------------

    def calculate_correlation(self, data1, data2) -> float:
        """Calcule la corrélation entre deux séries de données."""
        correlation_func = self.correlation_functions[self.correlation_type]
        try:
            corr_value, _ = correlation_func(data1, data2)
            return corr_value if not np.isnan(corr_value) else 0
        except (ValueError, TypeError) as e:
            logger.warning(f"Corrélation non calculable : {e}")
            return 0

    def calculate_all_correlations(self, dataset):
        """Calcule toutes les corrélations entre paires d'attributs du dataset."""
        attributes = list(dataset.columns)
        correlations = {}
        significant_correlations = {}

        logger.info(
            f"Calcul des corrélations ({self.correlation_type}) sur {len(attributes)} attributs "
            f"(seuil={self.correlation_threshold})"
        )

        for i in range(len(attributes)):
            for j in range(i + 1, len(attributes)):
                attr1, attr2 = attributes[i], attributes[j]
                corr_value = self.calculate_correlation(
                    dataset[attr1].values, dataset[attr2].values
                )
                correlations[(attr1, attr2)] = corr_value

                if abs(corr_value) >= self.correlation_threshold:
                    significant_correlations[(attr1, attr2)] = corr_value
                    logger.debug(f"{attr1} - {attr2}: {corr_value:.3f}")

        logger.info(
            f"Corrélations : {len(correlations)} calculées, "
            f"{len(significant_correlations)} significatives"
        )
        return correlations, significant_correlations

    def identify_relevant_attributes(self, significant_correlations) -> set:
        """Identifie les attributs participant à au moins une corrélation significative."""
        relevant_attributes = set()
        for attr1, attr2 in significant_correlations:
            relevant_attributes.add(attr1)
            relevant_attributes.add(attr2)

        logger.info(f"Attributs pertinents : {len(relevant_attributes)}")
        return relevant_attributes

    # ------------------------------------------------------------------
    #  DAG / support
    # ------------------------------------------------------------------


    def build_dag_structure(self, matrix) -> dict:
        """Construit la structure DAG (parents, enfants, racines, feuilles) à partir d'une matrice."""
        n = matrix.shape[0]
        parents = defaultdict(list)
        children = defaultdict(list)

        rows, cols = np.where(matrix == 1)
        for parent, child in zip(rows, cols):
            parents[child].append(parent)
            children[parent].append(child)

        leaves = [i for i in range(n) if len(children[i]) == 0]
        roots = [i for i in range(n) if len(parents[i]) == 0]

        return {
            "parents": dict(parents),
            "children": dict(children),
            "leaves": leaves,
            "roots": roots,
        }

    def compute_support_GGC(self, matrix, minsup: int = 2):
        """Calcule le support (longueur du plus long chemin) via un parcours topologique."""
        cache_key = matrix.tobytes()
        if cache_key in self.cache:
            return self.cache[cache_key]

        n = matrix.shape[0]
        dag = self.build_dag_structure(matrix)

        memory = np.zeros(n, dtype=np.int32)
        memory_chain = [[] for _ in range(n)]

        in_degree = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if i in dag["parents"]:
                in_degree[i] = len(dag["parents"][i])

        queue = deque()
        for root in dag["roots"]:
            memory[root] = 1
            memory_chain[root] = [root]
            queue.append(root)

        while queue:
            current = queue.popleft()
            if current in dag["children"]:
                for child in dag["children"][current]:
                    new_support = memory[current] + 1
                    if new_support > memory[child]:
                        memory[child] = new_support
                        memory_chain[child] = memory_chain[current] + [child]

                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        max_support = np.max(memory) if len(memory) > 0 else 0
        max_idx = np.argmax(memory) if len(memory) > 0 else 0
        best_chain = memory_chain[max_idx] if memory_chain[max_idx] else []
        result_chain = "->".join(str(e) for e in best_chain) if best_chain else ""

        result = (max_support, result_chain)
        self.cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # ÉTAPE 2 : 1-itemsets
    # ------------------------------------------------------------------

    def generate_1_itemsets_filtered(self, dataset, minsup, relevant_attributes):
        """Génère les 1-itemsets uniquement pour les attributs pertinents."""
        patterns = {}
        sorted_relevant = sorted(relevant_attributes)

        logger.info(f"Génération des 1-itemsets pour {len(sorted_relevant)} attributs pertinents")

        for attr in sorted_relevant:
            values = dataset[attr].values

            for sign, comparison in (("+", "greater"), ("-", "less")):
                matrix = generate_binary_matrix_vectorized(values, comparison)
                support = self.compute_support_GGC(matrix, minsup)

                if support[0] >= minsup:
                    patterns[f"{attr}{sign}"] = (matrix, support[0], support[1])
                    logger.debug(f"{attr}{sign} : support={support[0]} FRÉQUENT")
                else:
                    logger.debug(f"{attr}{sign} : support={support[0]} non fréquent")

        logger.info(f"{len(patterns)} 1-itemsets fréquents trouvés")
        return patterns

    # ------------------------------------------------------------------
    # ÉTAPE 3 : 2-itemsets
    # ------------------------------------------------------------------

    def generate_2_itemsets_from_correlations(
        self, dataset, frequent_1_itemsets, significant_correlations, minsup
    ):
        """Génère les 2-itemsets à partir des corrélations significatives."""
        patterns_2_itemsets = {}

        for (attr1, attr2), corr_value in significant_correlations.items():
            if corr_value > 0:
                candidates = [f"{attr1}+,{attr2}+", f"{attr1}-,{attr2}-"]
            else:
                candidates = [f"{attr1}+,{attr2}-", f"{attr1}-,{attr2}+"]

            for candidate in candidates:
                item1, item2 = candidate.split(",")

                if item1 in frequent_1_itemsets and item2 in frequent_1_itemsets:
                    matrix1 = frequent_1_itemsets[item1][0]
                    matrix2 = frequent_1_itemsets[item2][0]
                    combined_matrix = np.logical_and(matrix1, matrix2).astype(np.int8)
                    support = self.compute_support_GGC(combined_matrix, minsup)

                    if support[0] >= minsup:
                        normalized = self.normalize_pattern_order(candidate)
                        patterns_2_itemsets[normalized] = (combined_matrix, support[0], support[1])
                        logger.debug(f"{candidate} : support={support[0]} FRÉQUENT")

        logger.info(f"{len(patterns_2_itemsets)} 2-itemsets fréquents générés")
        return patterns_2_itemsets

    def normalize_pattern_order(self, pattern: str) -> str:
        """Normalise l'ordre des attributs dans un pattern (ex: 'B+,A-' -> 'A-,B+')."""
        items = pattern.split(",")
        return ",".join(sorted(items, key=lambda x: x[:-1]))

    # ------------------------------------------------------------------
    # ÉTAPE 4 : k-itemsets (k >= 3)
    # ------------------------------------------------------------------

    def generate_k_itemsets_from_correlations(
        self, dataset, frequent_k_minus_1, frequent_1_itemsets, k,
        significant_correlations, minsup, relevant_attributes
    ):
        """Génère les k-itemsets en étendant chaque (k-1)-itemset d'un nouvel attribut corrélé."""
        patterns_k_itemsets = {}
        attributes = list(relevant_attributes)

        for pattern_key, (matrix, _support, _chain) in frequent_k_minus_1.items():
            current_attributes = self.extract_attributes_from_pattern(pattern_key)
            candidate_attributes = [a for a in attributes if a not in current_attributes]

            for new_attr in candidate_attributes:
                correlations_valid = True
                new_attr_correlations = {}

                for existing_attr in current_attributes:
                    corr_value = significant_correlations.get(
                        (existing_attr, new_attr),
                        significant_correlations.get((new_attr, existing_attr)),
                    )

                    if corr_value is None or abs(corr_value) < self.correlation_threshold:
                        correlations_valid = False
                        break
                    new_attr_correlations[existing_attr] = corr_value

                if not correlations_valid:
                    continue

                possible_new_signs = self.generate_possible_signs(new_attr_correlations, pattern_key)

                for new_sign in possible_new_signs:
                    new_item = f"{new_attr}{new_sign}"
                    normalized_candidate = self.normalize_pattern_order(f"{pattern_key},{new_item}")

                    if not self.verify_all_subsets_present(normalized_candidate, frequent_k_minus_1, k):
                        continue
                    if new_item not in frequent_1_itemsets:
                        continue

                    combined_matrix = np.logical_and(
                        matrix, frequent_1_itemsets[new_item][0]
                    ).astype(np.int8)
                    support = self.compute_support_GGC(combined_matrix, minsup)

                    if support[0] >= minsup:
                        patterns_k_itemsets[normalized_candidate] = (
                            combined_matrix, support[0], support[1]
                        )
                        logger.debug(f"{normalized_candidate} : support={support[0]} FRÉQUENT")

        logger.info(f"{len(patterns_k_itemsets)} {k}-itemsets fréquents générés")
        return patterns_k_itemsets

    def extract_attributes_from_pattern(self, pattern: str) -> list:
        """Extrait les noms d'attributs d'un pattern (sans les signes)."""
        return [item[:-1] for item in pattern.split(",")]

    def generate_possible_signs(self, correlations_dict, current_pattern: str) -> list:
        """Détermine les signes compatibles pour un nouvel attribut, selon les corrélations existantes."""
        current_items = current_pattern.split(",")
        positive_compatible = True
        negative_compatible = True

        for item in current_items:
            attr, sign = item[:-1], item[-1]
            if attr not in correlations_dict:
                continue

            corr_value = correlations_dict[attr]
            if corr_value > 0:
                if sign == "+" and not positive_compatible:
                    positive_compatible = False
                if sign == "-" and not negative_compatible:
                    negative_compatible = False
            else:
                if sign == "+" and not negative_compatible:
                    negative_compatible = False
                if sign == "-" and not positive_compatible:
                    positive_compatible = False

        possible_signs = []
        if positive_compatible:
            possible_signs.append("+")
        if negative_compatible:
            possible_signs.append("-")

        return possible_signs if possible_signs else ["+", "-"]

    def verify_all_subsets_present(self, k_itemset: str, frequent_k_minus_1, k: int) -> bool:
        """Vérifie que tous les sous-ensembles de taille (k-1) du k-itemset sont fréquents."""
        items = k_itemset.split(",")
        for i in range(len(items)):
            subset = ",".join(items[:i] + items[i + 1:])
            if self.normalize_pattern_order(subset) not in frequent_k_minus_1:
                return False
        return True

    # ------------------------------------------------------------------
    # Nettoyage final
    # ------------------------------------------------------------------

    def is_complementary_pattern(self, pattern1: str, pattern2: str) -> bool:
        """Vérifie si deux patterns portent sur les mêmes attributs avec des signes opposés."""
        items1 = pattern1.split(",")
        items2 = pattern2.split(",")

        if len(items1) != len(items2):
            return False

        attrs1 = sorted([(item[:-1], item[-1]) for item in items1], key=lambda x: x[0])
        attrs2 = sorted([(item[:-1], item[-1]) for item in items2], key=lambda x: x[0])

        for (attr1, sign1), (attr2, sign2) in zip(attrs1, attrs2):
            if attr1 != attr2:
                return False
            if (sign1 == "+" and sign2 != "-") or (sign1 == "-" and sign2 != "+"):
                return False

        return True

    def remove_complementary_patterns(self, patterns: dict) -> dict:
        """Supprime les patterns complémentaires en gardant celui avec le plus de signes '+'."""
        clean_patterns = {}
        processed = set()
        pattern_list = list(patterns.items())

        for i, (pattern1, data1) in enumerate(pattern_list):
            if pattern1 in processed:
                continue

            complementary_found = False
            for pattern2, data2 in pattern_list[i + 1:]:
                if pattern2 in processed:
                    continue
                if self.is_complementary_pattern(pattern1, pattern2):
                    if pattern1.count("+") >= pattern2.count("+"):
                        clean_patterns[pattern1] = data1
                    else:
                        clean_patterns[pattern2] = data2
                    processed.update([pattern1, pattern2])
                    complementary_found = True
                    break

            if not complementary_found:
                clean_patterns[pattern1] = data1
                processed.add(pattern1)

        return clean_patterns

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def extract_gradual_patterns(self, dataset, minsup: int, max_size: int) -> dict:
        """
        Extrait les motifs graduels fréquents d'un dataset déjà prétraité.

        Parameters
        ----------
        dataset : pd.DataFrame
            Données numériques, sans valeurs manquantes ni doublons.
        minsup : int
            Support minimum pour qu'un motif soit considéré fréquent.
        max_size : int
            Taille maximale des itemsets à générer.

        Returns
        -------
        dict
            Motifs fréquents, sous la forme {pattern: (matrice, support, chaîne)}.
        """
        logger.info(
            f"Extraction GGC : minsup={minsup}, max_size={max_size}, "
            f"correlation={self.correlation_type} (seuil={self.correlation_threshold})"
        )

        _all_correlations, significant_correlations = self.calculate_all_correlations(dataset)
        if not significant_correlations:
            logger.warning("Aucune corrélation significative trouvée. Arrêt.")
            return {}

        relevant_attributes = self.identify_relevant_attributes(significant_correlations)
        if not relevant_attributes:
            logger.warning("Aucun attribut pertinent identifié. Arrêt.")
            return {}

        frequent_1_itemsets = self.generate_1_itemsets_filtered(dataset, minsup, relevant_attributes)
        if not frequent_1_itemsets:
            logger.warning("Aucun 1-itemset fréquent trouvé. Arrêt.")
            return {}

        frequent_2_itemsets = self.generate_2_itemsets_from_correlations(
            dataset, frequent_1_itemsets, significant_correlations, minsup
        )

        all_frequent_patterns = dict(frequent_1_itemsets)
        all_frequent_patterns.update(frequent_2_itemsets)

        if not frequent_2_itemsets:
            logger.warning("Aucun 2-itemset fréquent trouvé. Arrêt.")
            return all_frequent_patterns

        current_frequent = frequent_2_itemsets
        for k in range(3, max_size + 1):
            if not current_frequent:
                break

            frequent_k_itemsets = self.generate_k_itemsets_from_correlations(
                dataset, current_frequent, frequent_1_itemsets, k,
                significant_correlations, minsup, relevant_attributes
            )
            if not frequent_k_itemsets:
                break

            all_frequent_patterns.update(frequent_k_itemsets)
            current_frequent = frequent_k_itemsets

        logger.info(f"Patterns avant nettoyage : {len(all_frequent_patterns)}")
        clean_patterns = self.remove_complementary_patterns(all_frequent_patterns)
        logger.info(f"Patterns après nettoyage : {len(clean_patterns)}")

        return clean_patterns