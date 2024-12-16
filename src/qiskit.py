# from qiskit import QuantumCircuit, Aer, execute

# def measure_probability():
#     # Créer un circuit quantique avec 1 qubit
#     circuit = QuantumCircuit(1, 1)
#     circuit.h(0)  # Appliquer une porte Hadamard
#     circuit.measure(0, 0)  # Mesurer le qubit

#     # Exécuter le circuit sur le simulateur
#     simulator = Aer.get_backend('qasm_simulator')
#     result = execute(circuit, backend=simulator, shots=1024).result()
#     counts = result.get_counts(circuit)

#     # Calculer les probabilités
#     probabilities = {key: value / 1024 for key, value in counts.items()}
#     return probabilities

if __name__ == "__main__":
    probabilities = measure_probability()
    print("Probabilités mesurées:", probabilities)
