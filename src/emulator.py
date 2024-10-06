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


# Fonction pour afficher les probabilités
def print_results(probabilities, num_qubits):
    """
    Affiche les probabilités pour chaque état dans la base de calcul.
    """
    print(f"Probabilités finales pour les états |000...0> à |111...1> avec {num_qubits} qubits :")
    for i, prob in enumerate(probabilities):
        binary_state = f"|{i:0{num_qubits}b}>"
        print(f"{binary_state}: P = {prob:.4f}")


def binary_to_decimal(binary_string):
    """
    Convertit une chaîne binaire en nombre décimal.
    """
    return int(binary_string, 2)
