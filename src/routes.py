from flask import render_template, request, jsonify
from src.circuits import run_grover, run_deutsch_jozsa

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