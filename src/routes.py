from flask import Flask, render_template, request, jsonify, request
from src.gates import apply_gate
from src.emulator import measure_probabilities
from src.circuits import run_grover, run_deutsch_jozsa
from src.algorithms import solve_qubo
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World!"

if __name__ == "__main__":
    app.run()

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
        # Change from request.form to request.json for the entire request
        data = request.json  # Expecting JSON format for the entire request
        algorithm = data['algorithm']
        num_qubits = int(data['num_qubits'])
        
        if algorithm == 'Grover':
            target_state = data['target_state']
            if not all(bit in '01' for bit in target_state) or len(target_state) != num_qubits:
                return jsonify({'error': f"Please enter a valid binary target state with {num_qubits} bits."})
            result = run_grover(target_state)
        elif algorithm == 'Deutsch-Jozsa':
            result = run_deutsch_jozsa(num_qubits)
        elif algorithm == 'QUBO':
            Q = np.array(data['Q'])  # Expecting a 2D list for the QUBO matrix
            p = data.get('p', 1)  # Optional parameter for the number of layers
            solution, cost = solve_qubo(Q, p)
            return jsonify({'solution': solution.tolist(), 'cost': cost})
        elif algorithm == 'QAOA':
            Q = np.array(data['Q'])  # Expecting a 2D list for the QUBO matrix
            p = data.get('p', 1)  # Optional parameter for the number of layers
            solution, cost = solve_qubo(Q, p)  # Assuming you want to use the same function for QAOA
            return jsonify({'solution': solution.tolist(), 'cost': cost})
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

    @app.route('/quantum_emulator')
    def quantum_emulator():
        return render_template('quantum_emulator.html')

    @app.route('/quantum_walk')
    def quantum_walk():
        return render_template('quantum_walk.html')

    @app.route('/quantum_art')
    def quantum_art():
        return render_template('quantum_art.html')
