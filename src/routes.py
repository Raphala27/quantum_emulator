from flask import Flask, render_template, request, jsonify
from src.gates import apply_gate
from src.emulator import measure_probabilities
from src.circuits import run_grover, run_deutsch_jozsa
import numpy as np

def register_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/algorithms')
    def algorithms():
        return render_template('algorithms.html')

    @app.route('/custom_circuits')
    def custom_circuits():
        return render_template('custom_circuits.html')

    @app.route('/run_algorithm', methods=['POST'])
    def run_algorithm():
        algorithm = request.form['algorithm']
        num_qubits = int(request.form['num_qubits'])
        
        if algorithm == 'Grover':
            target_state = request.form['target_state']
            if not all(bit in '01' for bit in target_state) or len(target_state) != num_qubits:
                return jsonify({'error': f"Please enter a valid binary target state with {num_qubits} bits."})
            result = run_grover(target_state)
        elif algorithm == 'Deutsch-Jozsa':
            result = run_deutsch_jozsa(num_qubits)
        else:
            return jsonify({'error': 'Invalid algorithm selected.'})
        
        return jsonify({'result': result})
    
    @app.route('/run_circuit', methods=['POST'])
    def run_circuit():
        data = request.json
        num_qubits = data['num_qubits']
        circuit = data['circuit']

        # Initialize the quantum state
        state = np.zeros(2**num_qubits, dtype=complex)
        state[0] = 1  # Initialize to |0...0>

        # Apply gates
        for gate in circuit:
            if gate['type'] == 'CNOT':
                # For CNOT, we need to determine control and target qubits
                control = gate['qubit']
                target = (control + 1) % num_qubits  # Assuming CNOT applies to adjacent qubits
                state = apply_gate('CNOT', {'control': control, 'target': target}, state, num_qubits)
            else:
                state = apply_gate(gate['type'], {'qubit': gate['qubit']}, state, num_qubits)

        # Measure probabilities
        probabilities = measure_probabilities(state)

        # Format results
        results = []
        for i, prob in enumerate(probabilities):
            binary_state = f"{i:0{num_qubits}b}"
            results.append({
                'state': binary_state,
                'probability': float(prob)  # Convert to float for JSON serialization
            })

        return jsonify(results)