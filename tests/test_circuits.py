import unittest
import numpy as np

# Importe les fonctions depuis le script quantique
from src import *

class TestQuantumSimulator(unittest.TestCase):
    
    def test_hadamard_on_single_qubit(self):
        """Test de la porte Hadamard appliquée sur un seul qubit."""
        num_qubits = 1
        state = np.zeros(2 ** num_qubits, dtype=complex)
        state[0] = 1  # |0> état initial
        
        state_after_h = apply_single_qubit_gate(H, 0, state, num_qubits)
        expected_state = (1 / np.sqrt(2)) * np.array([1, 1])
        
        # Vérifie que l'état après l'application de H correspond à l'état attendu
        np.testing.assert_almost_equal(state_after_h, expected_state, decimal=6)
    
    def test_not_on_single_qubit(self):
        """Test de la porte NOT appliquée sur un seul qubit."""
        num_qubits = 1
        state = np.zeros(2 ** num_qubits, dtype=complex)
        state[0] = 1  # |0> état initial
        
        state_after_x = apply_single_qubit_gate(X, 0, state, num_qubits)
        expected_state = np.array([0, 1])  # |1>
        
        np.testing.assert_array_equal(state_after_x, expected_state)
    
    def test_cnot_on_two_qubits(self):
        """Test de la porte CNOT appliquée sur deux qubits."""
        num_qubits = 2
        state = np.zeros(2 ** num_qubits, dtype=complex)
        state[0] = 1  # |00> état initial
        
        # Appliquer NOT sur le premier qubit (pour avoir |10>)
        state = apply_single_qubit_gate(X, 0, state, num_qubits)
        state_after_cnot = apply_cnot(0, 1, state, num_qubits)  # CNOT avec qubit 0 comme contrôle
        
        expected_state = np.array([0, 0, 0, 1])  # Résultat attendu : |11>
        np.testing.assert_array_equal(state_after_cnot, expected_state)
    
    def test_measure_probabilities(self):
        """Test de la fonction de mesure des probabilités."""
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Superposition |0> et |1>
        probabilities = measure_probabilities(state)
        expected_probabilities = np.array([0.5, 0.5])
        
        np.testing.assert_almost_equal(probabilities, expected_probabilities, decimal=6)
    
    def test_hadamard_cnot_on_three_qubits(self):
        """Test d'un circuit à trois qubits avec Hadamard et CNOT."""
        num_qubits = 3
        state = np.zeros(2 ** num_qubits, dtype=complex)
        state[0] = 1  # |000> état initial
        
        # Appliquer Hadamard sur le qubit 0
        state = apply_single_qubit_gate(H, 0, state, num_qubits)
        
        # Appliquer NOT sur le qubit 1
        state = apply_single_qubit_gate(X, 1, state, num_qubits)
        
        # Appliquer CNOT entre qubit 0 (contrôle) et qubit 1 (cible)
        state = apply_cnot(0, 1, state, num_qubits)
        
        # Attendre que les probabilités correspondent à l'état de superposition attendu
        probabilities = measure_probabilities(state)
        
        expected_probabilities = np.array([
            0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0
        ])  # Probabilités des états |000> à |111> pour un circuit H, X et CNOT
        
        np.testing.assert_almost_equal(probabilities, expected_probabilities, decimal=6)

# Point d'entrée pour exécuter les tests
if __name__ == '__main__':
    unittest.main()
