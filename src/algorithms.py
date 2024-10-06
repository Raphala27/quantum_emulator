import numpy as np
from src.emulator import measure_probabilities
from src.gates import apply_single_qubit_gate, apply_cnot, H, X
from src.oracles import grover_oracle


# Algorithme de Deutsch-Jozsa
def deutsch_jozsa(oracle, num_qubits):
    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1

    for i in range(num_qubits):
        state = apply_single_qubit_gate(H, i, state, num_qubits)

    state = oracle(state, num_qubits)

    for i in range(num_qubits):
        state = apply_single_qubit_gate(H, i, state, num_qubits)

    probabilities = np.abs(state)**2
    measurement = np.argmax(probabilities)

    return measurement == 0


#TODO Téléportation quantique


def diffusion(n):
    """
    Crée l'opérateur de diffusion pour l'algorithme de Grover.
    
    :param n: Nombre de qubits
    :return: Matrice représentant l'opérateur de diffusion
    """
    s = np.ones(2**n) / np.sqrt(2**n)
    return 2 * np.outer(s, s) - np.eye(2**n)

def grover(n, target, iterations):
    """
    Implémente l'algorithme de Grover.
    
    :param n: Nombre de qubits
    :param target: État cible (sous forme d'entier)
    :param iterations: Nombre d'itérations de l'algorithme
    :return: État final après l'application de l'algorithme
    """
    state = np.ones(2**n) / np.sqrt(2**n)
    oracle_op = grover_oracle(n, target)
    diffusion_op = diffusion(n)
    
    for _ in range(iterations):
        state = oracle_op @ state
        state = diffusion_op @ state
    
    return state
