import numpy as np

# Fonction pour le produit tensoriel d'une liste de matrices
def tensor_product(matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result

# Mesure les probabilités des 2^n états possibles dans un système de n qubits
def measure_probabilities(state):
    probabilities = np.abs(state) ** 2  # Probabilités = carré des amplitudes
    return probabilities

