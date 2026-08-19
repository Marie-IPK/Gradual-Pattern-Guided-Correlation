"""
Algorithme GRITE pour l'extraction de motifs graduels fréquents.
Suppose un dataset déjà prétraité (voir src/data/processing.py).
"""

import itertools
import logging
import re

import numpy as np

from src.algorithms.common import generate_binary_matrix_vectorized
from src.utils import PerformanceMonitor

logger = logging.getLogger(__name__)


class GriteExtractor:
    """Extracteur de motifs graduels fréquents — algorithme GRITE."""

    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.last_pruned_count = None  # rempli après chaque appel à extract_gradual_patterns

    # ------------------------------------------------------------------
    # Génération des 1-itemsets
    # ------------------------------------------------------------------

    def generate_base_patterns(self, dataset, minsup):
        """
        Génère les 1-itemsets fréquents (un attribut, signe + ou -).

        BUGFIX (par rapport au code original) : matrix_moin était calculée
        avec la même comparaison que matrix_plus ("greater"), car le
        paramètre `sign` de generateMatrixOfSingle n'était en réalité
        jamais utilisé dans son corps. Les motifs "+" et "-" étaient donc
        identiques — ce qui va à l'encontre du but de l'algorithme
        (extraire des motifs croissants ET décroissants). Corrigé ici en
        utilisant explicitement comparison="less" pour le signe "-".
        """
        pattern = {}

        for attr in dataset.columns:
            matrix_plus = generate_binary_matrix_vectorized(dataset[attr].values, "greater")
            matrix_moin = generate_binary_matrix_vectorized(dataset[attr].values, "less")

            sup_plus = self.apply_support(matrix_plus)
            sup_moin = self.apply_support(matrix_moin)

            if sup_plus[0] >= minsup:
                pattern[attr + "+"] = (matrix_plus, sup_plus[0], sup_plus[1])

            # BUGFIX : le tuple (support, chaîne) était stocké tel quel
            # au lieu d'être déplié — corrigé pour rester cohérent avec
            # le format utilisé partout ailleurs : (matrice, support, chaîne)
            if sup_moin[0] >= minsup:
                pattern[attr + "-"] = (matrix_moin, sup_moin[0], sup_moin[1])

        return pattern

    # ------------------------------------------------------------------
    # Calcul du support GRITE (parcours récursif depuis chaque nœud)
    # ------------------------------------------------------------------

    def parents(self, son, matrix):
        """Trouve les parents (prédécesseurs) d'un nœud dans la matrice."""
        return [line for line in range(matrix.shape[0]) if matrix[line][son] == 1]

    def get_children(self, father, matrix):
        """Trouve les fils (successeurs) d'un nœud dans la matrice."""
        return [col for col in range(len(matrix[father])) if matrix[father][col] == 1]

    def _compute_support_from_node(self, begin, memory, matrix):
        """Calcule récursivement le support GRITE à partir d'un nœud donné."""
        children = self.get_children(begin, matrix)

        if not children:
            memory[begin][0] = 1
            return

        for child in children:
            if memory[child][0] == -1:
                self._compute_support_from_node(child, memory, matrix)

        for child in children:
            for parent in reversed(self.parents(child, matrix)):
                memory[parent][0] = max(memory[parent][0], memory[child][0] + 1)
                if child not in memory[parent][1]:
                    memory[parent][1].append(child)
                    memory[parent][1].append(parent)

    def apply_support(self, matrix):
        """Calcule le support GRITE global d'une matrice (meilleur chemin trouvé)."""
        results = []
        for i in range(matrix.shape[0]):
            memory = [[-1, []] for _ in range(matrix.shape[1])]
            self._compute_support_from_node(i, memory, matrix)
            results.append(max(memory))

        supports = [support for support, _chain in results]
        chains = [chain for _support, chain in results]
        result_str, _chain = self._chain_to_string((supports, chains))

        return max(supports), result_str

    def _chain_to_string(self, tuple_list):
        """Convertit (liste de supports, liste de chaînes) en la meilleure chaîne trouvée."""
        support_list, chaine_list = tuple_list
        best_idx = support_list.index(max(support_list))

        chaine = list(chaine_list[best_idx])
        chaine.reverse()

        chaine_dedup = []
        if chaine:
            chaine_dedup.append(chaine[0])
            for e in chaine:
                if e not in chaine_dedup:
                    chaine_dedup.append(e)

        result = "->".join(str(e) for e in chaine_dedup)
        return result, chaine_dedup

    # ------------------------------------------------------------------
    # Utilitaires ensemblistes
    # ------------------------------------------------------------------

    def find_subsets(self, s, n):
        """Renvoie tous les sous-ensembles de taille n."""
        return list(map(set, itertools.combinations(s, n)))

    def purify(self, list_set_string):
        """Supprime les combinaisons portant deux fois le même attribut (ex: 'A+, A-')."""
        dechet = []
        for elm in list_set_string:
            attrs = [x.replace("+", "").replace("-", "") for x in elm]
            if len(attrs) != len(set(attrs)):
                dechet.append(elm)
        for de in dechet:
            list_set_string.remove(de)

    def union_all(self, list_set):
        """Union de tous les ensembles d'une liste."""
        if len(list_set) <= 1:
            return list_set
        result = set()
        for s in list_set:
            result = result.union(s)
        return result

    def combine_matrices(self, matrices):
        """Combine une liste de matrices par produit élément par élément (AND logique)."""
        result = matrices[0]
        for m in matrices[1:]:
            result = result * m
        return result

    # ------------------------------------------------------------------
    # Extraction principale (k-itemsets, k >= 2)
    # ------------------------------------------------------------------

    def extract_gradual_patterns(self, dataset, minsup):
        """
        Extrait les motifs graduels fréquents avec l'algorithme GRITE.

        Le nombre de candidats élagués par l'optimisation de seuil est
        disponible après l'appel via self.last_pruned_count.

        Returns
        -------
        dict
            Motifs fréquents {pattern: (matrice, support, chaîne)}.
        """
        logger.info(f"Extraction GRITE : minsup={minsup}")

        frequent_pattern = self.generate_base_patterns(dataset, minsup)
        if not frequent_pattern:
            logger.warning("Aucun 1-itemset fréquent trouvé. Arrêt.")
            self.last_pruned_count = 0
            return {}

        new_patterns = {}
        pruned_count = 0
        frequents = set(frequent_pattern.keys())

        for num in range(2, len(dataset.columns) + 1):
            frequents = self.find_subsets(frequents, num)
            self.purify(frequents)

            for combo in frequents:
                str_pattern = " , ".join(combo)
                matrix = self.combine_matrices([frequent_pattern[x][0] for x in combo])

                if np.sum(matrix) > minsup * (minsup - 1) / 2:
                    support = self.apply_support(matrix)
                    if support[0] >= minsup:
                        new_patterns[str_pattern] = (matrix, support[0], support[1])
                else:
                    pruned_count += 1

            if not new_patterns:
                break

            frequent_pattern.update(new_patterns)
            frequents = self.union_all([set(re.split(" , ", k)) for k in new_patterns])
            new_patterns = {}

        self.last_pruned_count = pruned_count
        logger.info(f"GRITE : {len(frequent_pattern)} motifs fréquents, {pruned_count} élagués")
        return frequent_pattern

    # ------------------------------------------------------------------
    # Nettoyage final
    # ------------------------------------------------------------------

    def is_complementary_pattern(self, key1, key2):
        """Vérifie si deux motifs portent sur les mêmes attributs avec signes opposés."""
        if len(key1) != len(key2):
            return False

        flipped = key2.replace("+", "0").replace("-", "+").replace("0", "-")
        permutations = [
            " , ".join(p) for p in itertools.permutations(set(flipped.split(" , ")))
        ]
        return key1 in permutations

    def remove_complementary_patterns(self, patterns):
        """Supprime les motifs complémentaires, en gardant la variante '+' de chaque paire."""
        clean = {}
        keys = list(patterns.keys())
        to_delete = set()

        for i in range(len(keys)):
            if keys[i] in to_delete:
                continue
            for j in range(i + 1, len(keys)):
                if keys[j] in to_delete:
                    continue
                if self.is_complementary_pattern(keys[i], keys[j]):
                    to_delete.add(keys[i] if "-" in keys[i] else keys[j])
                    break

        for key, value in patterns.items():
            if key not in to_delete:
                clean[key] = value

        return clean