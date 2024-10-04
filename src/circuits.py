import numpy as np
from src.gates import apply_single_qubit_gate, apply_cnot, H, X, I
from src.algorithms import grover, deutsch_jozsa, quantum_teleportation, grover_oracle

# Fonction pour exécuter un circuit quantique manuel
def example_circuit_n_qubits(num_qubits):
    state = np.zeros(2 ** num_qubits, dtype=complex)
    state[0] = 1
    
    # Construction manuelle du circuit
    state = apply_single_qubit_gate(H, 0, state, num_qubits)
    state = apply_cnot(0, 1, state, num_qubits)
    state = apply_single_qubit_gate(X, 1, state, num_qubits)
    
    probabilities = measure_probabilities(state)
    
    # Afficher les résultats
    print(f"Probabilités finales pour les états |000...0> à |111...1> avec {num_qubits} qubits :")
    for i, prob in enumerate(probabilities):
        binary_state = f"|{i:0{num_qubits}b}>"
        print(f"{binary_state}: P = {prob:.4f}")


# Fonction pour exécuter un algorithme quantique prédéfini
def run_algorithm(algorithm_name, num_qubits):
    """
    Exécute un algorithme quantique prédéfini.
    - algorithm_name: nom de l'algorithme (ex: 'grover', 'deutsch-jozsa', 'quantum_teleportation').
    - num_qubits: nombre de qubits dans le système.
    """
    if algorithm_name == 'grover':
        state = grover(num_qubits)
    elif algorithm_name == 'deutsch-jozsa':
        state = deutsch_jozsa(num_qubits)
    elif algorithm_name == 'quantum_teleportation':
        state = quantum_teleportation(num_qubits)
    else:
        raise ValueError(f"L'algorithme {algorithm_name} n'est pas implémenté.")
    
    # Mesurer les probabilités des 2^n états
    probabilities = measure_probabilities(state)
    
    # Afficher les résultats
    print(f"Résultats pour l'algorithme {algorithm_name} avec {num_qubits} qubits :")
    for i, prob in enumerate(probabilities):
        binary_state = f"|{i:0{num_qubits}b}>"
        print(f"{binary_state}: P = {prob:.4f}")


# Mesurer les probabilités des 2^n états possibles dans un système de n qubits
def measure_probabilities(state):
    probabilities = np.abs(state) ** 2
    return probabilities


# Choix entre la construction manuelle du circuit ou l'exécution d'un algorithme prédéfini
def main():
    print("Voulez-vous construire un circuit à la main ou exécuter un algorithme préconstruit ?")
    choice = input("Tapez 'manuel' pour construire un circuit, ou 'algo' pour exécuter un algorithme: ")

    if choice == 'manuel':
        num_qubits = int(input("Combien de qubits dans votre circuit ? "))
        example_circuit_n_qubits(num_qubits)
    elif choice == 'algo':
        algo = input("Quel algorithme voulez-vous exécuter ('grover', 'deutsch-jozsa', 'bernstein-vazirani') ? ")
        num_qubits = int(input("Combien de qubits dans votre circuit ? "))
        run_algorithm(algo, num_qubits)
    else:
        print("Choix non valide.")


if __name__ == "__main__":
    main()
