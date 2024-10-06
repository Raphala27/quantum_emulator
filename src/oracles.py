import numpy as np
from src.gates import apply_single_qubit_gate, apply_cnot, X, H


def deutsch_jozsa_oracle(state, num_qubits, oracle_type):
    """
    Oracle pour l'algorithme de Deutsch-Jozsa.
    Applique une fonction constante ou équilibrée à l'état quantique.
    - state: l'état quantique initial.
    - num_qubits: nombre de qubits dans le registre.
    - oracle_type: "constant" ou "équilibré", détermine la nature de l'oracle.
    """
    oracle_matrix = np.eye(2 ** num_qubits, dtype=complex)

    if oracle_type == "constant":
        # Oracle constant : ne modifie pas l'état (fonction f(x) = 0 ou f(x) = 1 pour tous les x)
        # Cela équivaut à ne rien faire, ou à appliquer une porte X sur un qubit de sortie fixe pour toujours renvoyer 1
        pass  # Ici, l'oracle constant ne modifie pas l'état du tout

    elif oracle_type == "équilibré":
        # Oracle équilibré : inverse la phase pour la moitié des états
        for i in range(2 ** (num_qubits - 1), 2 ** num_qubits):
            oracle_matrix[i, i] = -1  # Inversion de phase sur la moitié des états

    else:
        raise ValueError("Le type d'oracle doit être 'constant' ou 'équilibré'.")

    return np.dot(oracle_matrix, state)


# Oracle pour Grover : Marque une solution (par exemple, la solution "11")
def grover_oracle(n, target):
    """
    Crée un oracle pour l'algorithme de Grover.
    
    :param n: Nombre de qubits
    :param target: État cible (sous forme d'entier)
    :return: Matrice représentant l'oracle
    """
    oracle = np.eye(2**n)
    oracle[target, target] = -1
    return oracle
