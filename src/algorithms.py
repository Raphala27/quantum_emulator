import numpy as np
from scipy.optimize import minimize
from src.emulator import measure_state
from src.gates import apply_cnot_qaoa, apply_rx_gate, apply_rz_gate, apply_single_qubit_gate, H, create_quantum_state
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


def classical_cost_function(Q, binary_string):
    """
    Calcule le coût classique pour une solution binaire donnée.
    
    :param Q: Matrice QUBO
    :param binary_string: Chaîne binaire représentant une solution
    :return: Valeur du coût
    """
    x = np.array([int(bit) for bit in binary_string])
    return np.dot(x, np.dot(Q, x))


def create_qaoa_circuit(Q, beta, gamma):
    n_qubits = Q.shape[0]
    state = create_quantum_state(n_qubits)
    
    # Initialiser à une superposition égale
    for i in range(n_qubits):
        state = apply_rx_gate(state, i, np.pi / 2)

    # Appliquer l'Hamiltonien du problème
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if Q[i][j] != 0:
                state = apply_cnot_qaoa(state, i, j)
                state = apply_rz_gate(state, j, 2 * gamma * Q[i][j])
                state = apply_cnot_qaoa(state, i, j)
        state = apply_rz_gate(state, i, gamma * Q[i][i])

    # Appliquer l'Hamiltonien de mixage
    for i in range(n_qubits):
        state = apply_rx_gate(state, i, 2 * beta)

    return state

def qaoa_cost_function(Q, params):
    n_qubits = Q.shape[0]
    p = len(params) // 2
    beta = params[:p]
    gamma = params[p:]

    state = create_quantum_state(n_qubits)
    for b, g in zip(beta, gamma):
        state = create_qaoa_circuit(Q, b, g)

    exp_value = 0
    for _ in range(1000):  # nombre de mesures
        measured_state = measure_state(state)
        exp_value += classical_cost_function(Q, measured_state)
    
    return exp_value / 1000

def solve_qubo(Q, p=1):
    initial_params = np.random.rand(2 * p) * 2 * np.pi
    result = minimize(lambda params: qaoa_cost_function(Q, params), initial_params, method='L-BFGS-B')

    best_params = result.x
    beta = best_params[:p]
    gamma = best_params[p:]

    final_state = create_quantum_state(Q.shape[0])
    for b, g in zip(beta, gamma):
        final_state = create_qaoa_circuit(Q, b, g)

    best_solution = None
    best_cost = float('inf')
    for _ in range(10000):  # Augmenter le nombre de mesures
        measured_state = measure_state(final_state)
        cost = classical_cost_function(Q, measured_state)
        if cost < best_cost:
            best_cost = cost
            best_solution = measured_state

    return best_solution, best_cost


Q = np.array([[-1, 2], [2, -1]])
Q1 = np.array([[0, -1, -1],
               [-1, 0, -1],
               [-1, -1, 0]])
Q2 = np.array([
    [5, 0, 0, -5, -3, 0],
    [0, 5, 0, -5, 0, -3],
    [0, 0, 5, 0, -4, -4],
    [-5, -5, 0, 5, 0, 0],
    [-3, 0, -4, 0, 5, 0],
    [0, -3, -4, 0, 0, 5]
])

solution, cost = solve_qubo(Q2, p=2)
print(f"Meilleure solution trouvée finale : {solution}")
print(f"Coût de la solution : {cost}")