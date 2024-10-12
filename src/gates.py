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

# New funtions



# Add this new function
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
