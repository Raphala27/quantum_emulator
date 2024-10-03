import numpy as np

from src import *


# Fonction pour exécuter un exemple de circuit quantique à n qubits
def example_circuit_n_qubits(num_qubits):
    # État initial |000...0> dans l'espace de Hilbert de num_qubits
    state = np.zeros(2 ** num_qubits, dtype=complex)
    state[0] = 1  # Initialise à |000...0>
    
    # Appliquer des portes
    state = apply_single_qubit_gate(H, 0, state, num_qubits)  # Hadamard sur le qubit 0
    state = apply_cnot(0, 1, state, num_qubits)               # CNOT entre qubit 0 (control) et qubit 1 (target)
    state = apply_single_qubit_gate(X, 1, state, num_qubits)  # NOT sur le qubit 1
    # state = apply_single_qubit_gate(H, 2, state, num_qubits)  # Hadamard sur le qubit 0
    # state = apply_single_qubit_gate(X, 2, state, num_qubits)  # NOT sur le qubit 1
    # state = apply_single_qubit_gate(X, 3, state, num_qubits)  # NOT sur le qubit 1
    # state = apply_single_qubit_gate(H, 4, state, num_qubits)  # Hadamard sur le qubit 0
    # state = apply_single_qubit_gate(X, 5, state, num_qubits)  # NOT sur le qubit 1
    # state = apply_cnot(4, 5, state, num_qubits)               # CNOT entre qubit 0 (control) et qubit 1 (target)
    # state = apply_single_qubit_gate(H, 5, state, num_qubits)  # Hadamard sur le qubit 0
    # if num_qubits > 2:
    #     state = apply_single_qubit_gate(H, num_qubits - 1, state, num_qubits)  # Hadamard sur le dernier qubit

    # Mesurer les probabilités des 2^n états
    probabilities = measure_probabilities(state)
    
    # Afficher les résultats
    print(f"Probabilités finales pour les états |000...0> à |111...1> avec {num_qubits} qubits :")
    for i, prob in enumerate(probabilities):
        binary_state = f"|{i:0{num_qubits}b}>"
        print(f"{binary_state}: P = {prob:.4f}")

# Exécution de l'exemple pour un circuit à n qubits
example_circuit_n_qubits(3)  # Remplacer 3 par le nombre de qubits souhaité
