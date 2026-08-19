""" This fonction is the same for ggc.py and grite ans sgrite.py"""
import numpy as np


def generate_binary_matrix_vectorized(values, comparison: str = "greater"):
    """
    Génère une matrice binaire de comparaison entre toutes les paires de valeurs,
    de manière vectorisée (remplace les doubles boucles Python).

    matrix[i][j] = 1 si values[j] > values[i] (comparison='greater')
                   ou values[j] < values[i] (comparison='less'),
                   avec suppression de la diagonale et règle d'antisymétrie.
    """
    values = np.array(values)
    values_i = values[:, np.newaxis]
    values_j = values[np.newaxis, :]

    if comparison == "greater":
        matrix = (values_j > values_i).astype(np.int8)
    else:
        matrix = (values_j < values_i).astype(np.int8)

    np.fill_diagonal(matrix, 0)
    antisym_mask = matrix == 1
    matrix[antisym_mask.T] = 0

    return matrix