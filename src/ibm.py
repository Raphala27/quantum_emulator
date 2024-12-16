from qiskit import QuantumCircuit, transpile, assemble, execute
from qiskit.providers.ibmq import IBMQ

def run_on_ibm():
    # Charger votre compte IBMQ
    IBMQ.load_account()  # Assurez-vous d'avoir configuré votre compte IBMQ

    # Créer un circuit quantique
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)  # Appliquer une porte Hadamard
    circuit.measure(0, 0)  # Mesurer le qubit

    # Choisir un backend
    backend = IBMQ.get_provider(hub='ibm-q').get_backend('ibmq_qasm_simulator')  # Remplacez par un vrai backend

    # Compiler et exécuter le circuit
    transpiled_circuit = transpile(circuit, backend)
    qobj = assemble(transpiled_circuit)
    job = execute(circuit, backend, shots=1024)
    result = job.result()

    # Obtenir les résultats
    counts = result.get_counts(circuit)
    print("Résultats:", counts)

if __name__ == "__main__":
    run_on_ibm()
