from flask import Flask, render_template, request, jsonify
import numpy as np
import io
from src.emulator import binary_to_decimal, measure_probabilities
from src.algorithms import grover, deutsch_jozsa
import sys

from src.gates import H, X, apply_cnot, apply_single_qubit_gate



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


# New GUI-based main function
# Modify run_grover and run_deutsch_jozsa to return their output as a string
def run_grover(target_state_binary):
    output = io.StringIO()
    sys.stdout = output

    n_qubits = len(target_state_binary)
    target_state = binary_to_decimal(target_state_binary)
    
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

    sys.stdout = sys.__stdout__
    return output.getvalue()

def run_deutsch_jozsa(num_qubits):
    output = io.StringIO()
    sys.stdout = output

    print(f"Exécution de l'algorithme de Deutsch-Jozsa avec {num_qubits} qubits")
    
    choice = np.random.choice(["c", "e"])
    
    if choice == "c":
        constant_value = np.random.choice([0, 1])
        oracle = lambda state, n: state * (-1)**constant_value
        print("Fonction choisie : constante")
    else:
        oracle = lambda state, n: state * np.array([(-1)**(bin(i).count('1') % 2) for i in range(2**n)])
        print("Fonction choisie : équilibrée")

    result = deutsch_jozsa(oracle, num_qubits)
    
    print(f"Résultat : La fonction est {'constante' if result else 'équilibrée'}")

    sys.stdout = sys.__stdout__
    return output.getvalue()

# # New GUI-based main function
# def main():
#     root = tk.Tk()
#     root.title("Quantum Algorithm Simulator")
#     root.geometry("400x300")

#     frame = ttk.Frame(root, padding="10")
#     frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

#     algorithm_var = tk.StringVar()
#     num_qubits_var = tk.StringVar()
#     target_state_var = tk.StringVar()

#     ttk.Label(frame, text="Select Algorithm:").grid(column=0, row=0, sticky=tk.W, pady=5)
#     algorithm_combo = ttk.Combobox(frame, textvariable=algorithm_var)
#     algorithm_combo['values'] = ('Grover', 'Deutsch-Jozsa')
#     algorithm_combo.grid(column=1, row=0, sticky=(tk.W, tk.E), pady=5)
#     algorithm_combo.current(0)

#     ttk.Label(frame, text="Number of Qubits:").grid(column=0, row=1, sticky=tk.W, pady=5)
#     ttk.Entry(frame, textvariable=num_qubits_var).grid(column=1, row=1, sticky=(tk.W, tk.E), pady=5)

#     ttk.Label(frame, text="Target State (Grover):").grid(column=0, row=2, sticky=tk.W, pady=5)
#     ttk.Entry(frame, textvariable=target_state_var).grid(column=1, row=2, sticky=(tk.W, tk.E), pady=5)

#     def run_selected_algorithm():
#         algorithm = algorithm_var.get()
#         try:
#             num_qubits = int(num_qubits_var.get())
#         except ValueError:
#             messagebox.showerror("Error", "Please enter a valid number of qubits.")
#             return

#         if algorithm == "Grover":
#             target_state = target_state_var.get()
#             if not all(bit in '01' for bit in target_state) or len(target_state) != num_qubits:
#                 messagebox.showerror("Error", f"Please enter a valid binary target state with {num_qubits} bits.")
#                 return
#             result = run_grover(target_state)
#         elif algorithm == "Deutsch-Jozsa":
#             result = run_deutsch_jozsa(num_qubits)
#         else:
#             messagebox.showerror("Error", "Please select a valid algorithm.")
#             return

#         # Print result to console
#         print(result)

#         # Show result in messagebox
#         messagebox.showinfo("Algorithm Result", result)

#     ttk.Button(frame, text="Run Algorithm", command=run_selected_algorithm).grid(column=0, row=3, columnspan=2, pady=20)

#     root.mainloop()

# if __name__ == "__main__":


