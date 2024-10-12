import numpy as np

from src.emulator import tensor_product


# Définition des portes quantiques de base
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)
S = np.array([[1, 0], [0, 1j]])
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])


# Appliquer une porte à un qubit spécifique dans un état de n qubits
def apply_single_qubit_gate(gate, qubit_index, state, num_qubits):
    gates = [I] * num_qubits
    gates[qubit_index] = gate
    full_gate = tensor_product(gates)
    return np.dot(full_gate, state)


def apply_two_qubit_gate(gate, control_qubit, target_qubit, state, num_qubits):
    """ Applique une porte à deux qubits dans un état quantique. """
    full_gate = np.eye(2 ** num_qubits)
    for i in range(2 ** (num_qubits - 2)):
        indices = np.array([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
        full_gate[np.ix_(indices, indices)] = gate
    return np.dot(full_gate, state)

def apply_two_qubit_gate(gate, control_qubit, target_qubit, state, num_qubits):
    """ Applique une porte à deux qubits dans un état quantique. """
    full_gate = np.eye(2 ** num_qubits)
    for i in range(2 ** (num_qubits - 2)):
        indices = np.array([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
        full_gate[np.ix_(indices, indices)] = gate
    return np.dot(full_gate, state)


def apply_cnot(target, control, state, num_qubits):

    cnot = np.eye(2 ** num_qubits, dtype=complex)
    for i in range(2 ** num_qubits):
        control_bit = (i >> (num_qubits - control - 1)) & 1
        target_bit = (i >> (num_qubits - target - 1)) & 1
        if control_bit == 1:
            flipped_state = i ^ (1 << (num_qubits - target - 1))
            cnot[i, i] = 0
            cnot[i, flipped_state] = 1
    
    return np.dot(cnot, state)


def apply_cnot_qaoa(state, control_qubit, target_qubit):
    """ Applique une porte CNOT entre deux qubits. """
    num_qubits = int(np.log2(len(state)))
    cnot_matrix = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]])
    
    return apply_two_qubit_gate(cnot_matrix, control_qubit, target_qubit, state, num_qubits)



def apply_cnot_qaoa(state, control_qubit, target_qubit):
    """ Applique une porte CNOT entre deux qubits. """
    num_qubits = int(np.log2(len(state)))
    cnot_matrix = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]])
    
    return apply_two_qubit_gate(cnot_matrix, control_qubit, target_qubit, state, num_qubits)


def apply_gate(gate_type, params, state, num_qubits):
    if gate_type == 'H':
        return apply_single_qubit_gate(H, params['qubit'], state, num_qubits)
    elif gate_type == 'CNOT':
        return apply_cnot(params['control'], params['target'], state, num_qubits)
    elif gate_type == 'X':
        return apply_single_qubit_gate(X, params['qubit'], state, num_qubits)    
    elif gate_type == 'Y':
        return apply_single_qubit_gate(Y, params['qubit'], state, num_qubits)
    elif gate_type == 'Z':
        return apply_single_qubit_gate(Z, params['qubit'], state, num_qubits)
    elif gate_type == 'I':
        return apply_single_qubit_gate(I, params['qubit'], state, num_qubits)
    elif gate_type == 'S':
        return apply_single_qubit_gate(S, params['qubit'], state, num_qubits)
    elif gate_type == 'T':
        return apply_single_qubit_gate(T, params['qubit'], state, num_qubits)

    else:
        raise ValueError(f"Unknown gate type: {gate_type}")


def apply_rx_gate(state, qubit_index, theta):
    """
    Applique une porte Rx sur un qubit donné dans l'état quantique.

    :param state: État quantique actuel
    :param qubit_index: Index du qubit sur lequel appliquer la porte
    :param theta: Angle de rotation pour la porte Rx
    :return: Nouvel état quantique après application de la porte Rx
    """
    Rx = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                   [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
    
    return apply_single_qubit_gate(Rx, qubit_index, state, int(np.log2(len(state))))


def apply_rz_gate(state, qubit_index, theta):
    """
    Applique une porte Rz sur un qubit donné dans l'état quantique.

    :param state: État quantique actuel
    :param qubit_index: Index du qubit sur lequel appliquer la porte
    :param theta: Angle de rotation pour la porte Rz
    :return: Nouvel état quantique après application de la porte Rz
    """
    Rz = np.array([[np.exp(-1j * theta / 2), 0],
                   [0, np.exp(1j * theta / 2)]])
    
    return apply_single_qubit_gate(Rz, qubit_index, state, int(np.log2(len(state))))


def create_quantum_state(num_qubits):
    """
    Crée un état quantique initial dans une superposition égale de tous les états possibles.
    
    :param num_qubits: Nombre de qubits
    :return: Vecteur d'état quantique initial
    """
    # Créer un état de base |0>...|0> pour tous les qubits
    state = np.zeros(2**num_qubits)
    state[0] = 1  # L'état |00...0> est initialisé

    # Appliquer la porte Hadamard à chaque qubit pour obtenir une superposition
    for i in range(num_qubits):
        state = apply_single_qubit_gate(H, i, state, num_qubits)
    return state

def apply_rx_gate(state, qubit_index, theta):
    """
    Applique une porte Rx sur un qubit donné dans l'état quantique.

    :param state: État quantique actuel
    :param qubit_index: Index du qubit sur lequel appliquer la porte
    :param theta: Angle de rotation pour la porte Rx
    :return: Nouvel état quantique après application de la porte Rx
    """
    Rx = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                   [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
    
    return apply_single_qubit_gate(Rx, qubit_index, state, int(np.log2(len(state))))


def apply_rz_gate(state, qubit_index, theta):
    """
    Applique une porte Rz sur un qubit donné dans l'état quantique.

    :param state: État quantique actuel
    :param qubit_index: Index du qubit sur lequel appliquer la porte
    :param theta: Angle de rotation pour la porte Rz
    :return: Nouvel état quantique après application de la porte Rz
    """
    Rz = np.array([[np.exp(-1j * theta / 2), 0],
                   [0, np.exp(1j * theta / 2)]])
    
    return apply_single_qubit_gate(Rz, qubit_index, state, int(np.log2(len(state))))


def create_quantum_state(num_qubits):
    """
    Crée un état quantique initial dans une superposition égale de tous les états possibles.
    
    :param num_qubits: Nombre de qubits
    :return: Vecteur d'état quantique initial
    """
    # Créer un état de base |0>...|0> pour tous les qubits
    state = np.zeros(2**num_qubits)
    state[0] = 1  # L'état |00...0> est initialisé

    # Appliquer la porte Hadamard à chaque qubit pour obtenir une superposition
    for i in range(num_qubits):
        state = apply_single_qubit_gate(H, i, state, num_qubits)
    return state