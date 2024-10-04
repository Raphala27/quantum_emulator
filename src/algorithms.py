import numpy as np
from src.gates import apply_single_qubit_gate, apply_cnot, H, X


# Algorithme de Deutsch-Jozsa
def deutsch_jozsa(oracle, num_qubits):
    """
    Implémente l'algorithme de Deutsch-Jozsa pour déterminer si une fonction f est constante ou équilibrée.
    - oracle: fonction qui représente l'oracle (un circuit quantique) appliqué au registre de qubits.
    - num_qubits: nombre de qubits du registre.

    Retourne True si la fonction est constante, False si elle est équilibrée.
    """
    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1

    for i in range(num_qubits):
        state = apply_single_qubit_gate(H, i, state, num_qubits)

    # Appliquer l'oracle (la fonction f) au registre de qubits
    state = oracle(state, num_qubits)

    for i in range(num_qubits):
        state = apply_single_qubit_gate(H, i, state, num_qubits)

    probabilities = np.abs(state)**2
    measurement = np.argmax(probabilities)

    # Si le résultat de la mesure est |000...0>, la fonction est constante, sinon elle est équilibrée.
    return measurement == 0


# Algorithme de Grover
def grover(oracle, num_qubits, iterations):
    """
    Implémente l'algorithme de Grover pour rechercher dans une base de données non triée.
    - oracle: fonction oracle qui marque la solution correcte.
    - num_qubits: nombre de qubits du registre.
    - iterations: nombre d'itérations de l'algorithme (proche de sqrt(2^n)).
    
    Retourne l'état final après les itérations de l'algorithme.
    """
    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1

    for i in range(num_qubits):
        state = apply_single_qubit_gate(H, i, state, num_qubits)

    for _ in range(iterations):
        state = oracle(state, num_qubits)

        for i in range(num_qubits):
            state = apply_single_qubit_gate(H, i, state, num_qubits)

        # Appliquer la diffusion d'inversion autour de |s> (diffusion d'amplitude)
        state = apply_inversion_about_mean(state, num_qubits)

        for i in range(num_qubits):
            state = apply_single_qubit_gate(H, i, state, num_qubits)

    return state


# Inversion autour de la moyenne (nécessaire pour Grover)
def apply_inversion_about_mean(state, num_qubits):
    """
    Implémente l'inversion autour de la moyenne pour l'algorithme de Grover.
    Applique la transformation |psi> -> 2|s><s| - I à l'état quantique donné.
    """
    # État uniforme |s> = (1/sqrt(2^n)) * sum(|x>)
    uniform_state = np.ones(2**num_qubits, dtype=complex) / np.sqrt(2**num_qubits)

    # Diffusion d'amplitude : 2|s><s| - I
    return 2 * np.dot(uniform_state, state) * uniform_state - state


def grover_oracle(state, num_qubits):
    """
    Oracle pour l'algorithme de Grover.
    Inverse la phase de l'état cible (par exemple |11...1>).
    """
    # Cible l'état |11...1> dans l'espace de Hilbert (dernier état possible)
    target_state = np.zeros(2 ** num_qubits, dtype=complex)
    target_state[-1] = 1  # Cible l'état |11...1>

    # Appliquer une porte Z pour inverser la phase de l'état cible
    return state - 2 * np.dot(target_state, state) * target_state


# Téléportation quantique
def quantum_teleportation():
    """
    Implémente l'algorithme de téléportation quantique.
    Téléporte un qubit à un autre qubit distant en utilisant un état de Bell.
    """
    # Initialisation de l'état de Bell entre Alice et Bob
    state = np.zeros(4, dtype=complex)
    state[0] = 1  # Initialisation |00>

    state = apply_single_qubit_gate(H, 0, state, 2)

    state = apply_cnot(0, 1, state, 2)

    # Alice applique une mesure sur ses qubits
    measurement_result = np.argmax(np.abs(state)**2)

    if measurement_result == 1:
        state = apply_single_qubit_gate(X, 1, state, 2)

    return state
