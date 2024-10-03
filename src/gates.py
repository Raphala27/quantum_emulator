import numpy as np

from src.emulator import tensor_product

# Définition des portes quantiques de base
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])  # Hadamard
X = np.array([[0, 1], [1, 0]])  # NOT / Pauli-X
I = np.eye(2)  # Identité pour un qubit

# Appliquer une porte à un qubit spécifique dans un état de n qubits
def apply_single_qubit_gate(gate, qubit_index, state, num_qubits):
    gates = [I] * num_qubits
    gates[qubit_index] = gate
    full_gate = tensor_product(gates)
    return np.dot(full_gate, state)

# Appliquer une porte CNOT sur deux qubits (control et target)
def apply_cnot(control, target, state, num_qubits):
    # Construire la porte CNOT pour l'espace de Hilbert complet de n qubits
    cnot = np.eye(2 ** num_qubits, dtype=complex)
    for i in range(2 ** num_qubits):
        control_bit = (i >> (num_qubits - control - 1)) & 1  # Extraire le bit de contrôle
        target_bit = (i >> (num_qubits - target - 1)) & 1    # Extraire le bit cible
        if control_bit == 1:
            flipped_state = i ^ (1 << (num_qubits - target - 1))  # Inverser le bit cible
            cnot[i, i] = 0
            cnot[i, flipped_state] = 1
    
    return np.dot(cnot, state)