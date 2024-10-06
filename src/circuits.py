import numpy as np
from src.emulator import binary_to_decimal, measure_probabilities, print_results
from src.gates import apply_single_qubit_gate, apply_cnot, H, X
from src.algorithms import grover, deutsch_jozsa

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


def run_grover(target_state_binary):
    """
    Exécute l'algorithme de Grover avec un état cible spécifié en binaire.
    
    :param target_state_binary: État cible sous forme de chaîne binaire (ex: "01", "0111")
    """
    n_qubits = len(target_state_binary)
    target_state = binary_to_decimal(target_state_binary)
    
    # Calcul du nombre optimal d'itérations
    iterations = int(np.floor(np.pi/4 * np.sqrt(2**n_qubits)))
    
    result = grover(n_qubits, target_state, iterations)
    
    print(f"Nombre de qubits : {n_qubits}")
    print(f"État cible : {target_state_binary} (décimal: {target_state})")
    print(f"Nombre d'itérations : {iterations}")
    print("\nProbabilités de mesure :")
    
    probabilities = measure_probabilities(result)
    for i, prob in enumerate(probabilities):
        binary = format(i, f'0{n_qubits}b')
        print(f"|{binary}⟩ : {prob:.4f}")
    
    most_probable_state = format(np.argmax(probabilities), f'0{n_qubits}b')
    print(f"\nÉtat le plus probable : |{most_probable_state}⟩")
    print(f"Probabilité de l'état le plus probable : {np.max(probabilities):.4f}")


def run_deutsch_jozsa(num_qubits):
    print(f"Exécution de l'algorithme de Deutsch-Jozsa avec {num_qubits} qubits")
    
    # Demander à l'utilisateur de choisir entre une fonction constante et une fonction équilibrée
    choice = input("Choisissez le type de fonction (constante (c)/équilibrée (e)): ").lower()
    
    if choice == "c":
        # Oracle pour une fonction constante (toujours 0 ou toujours 1)
        constant_value = np.random.choice([0, 1])
        oracle = lambda state, n: state * (-1)**constant_value
    elif choice == "e":
        # Oracle pour une fonction équilibrée (0 pour la moitié des entrées, 1 pour l'autre moitié)
        oracle = lambda state, n: state * np.array([(-1)**(bin(i).count('1') % 2) for i in range(2**n)])
    else:
        print("Choix non valide. Utilisation d'une fonction constante par défaut.")
        oracle = lambda state, n: state

    result = deutsch_jozsa(oracle, num_qubits)
    
    print(f"Résultat : La fonction est {'constante' if result else 'équilibrée'}")


# Fonction pour exécuter un algorithme quantique prédéfini
def run_algorithm(algorithm_name, num_qubits):
    """
    Exécute un algorithme quantique prédéfini.
    - algorithm_name: nom de l'algorithme (ex: 'grover', 'deutsch-jozsa').
    - num_qubits: nombre de qubits dans le système.
    """
    state = None

    if algorithm_name == 'g':
        target_state_binary = input("Entrez l'état cible en notation binaire (ex: 01, 0111) : ")
        state = run_grover(target_state_binary)
        
    elif algorithm_name == 'dj':
        run_deutsch_jozsa(num_qubits)

    else:
        raise ValueError(f"L'algorithme {algorithm_name} n'est pas implémenté.")


# Choix entre la construction manuelle du circuit ou l'exécution d'un algorithme prédéfini
def main():
    print("Choisissez l'algorithme quantique à exécuter :")
    choice = input("Tapez 'g --> grover', 'dj --> deutsch-jozsa': ")

    if choice in ['g', 'dj']:
        num_qubits = int(input("Combien de qubits dans votre circuit ? "))
        run_algorithm(choice, num_qubits)
    else:
        print("Choix non valide.")


if __name__ == "__main__":
    main()
