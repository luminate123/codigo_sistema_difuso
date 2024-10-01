from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)
CORS(app, resources={r"/dengue_risk": {"origins": "http://127.0.0.1:5500"}})

# Define the fuzzy logic system
fever = ctrl.Antecedent(np.arange(35.9, 43, 0.1), 'fever')
symptoms = ctrl.Antecedent(np.arange(0, 11, 1), 'symptoms')
platelets = ctrl.Antecedent(np.arange(50, 451, 1), 'platelets')
dengue_risk = ctrl.Consequent(np.arange(0, 101, 1), 'dengue_risk')

# Define the linguistic labels and membership functions
fever['Normal'] = fuzz.trimf(fever.universe, [35, 37, 38])
fever['Alta'] = fuzz.trimf(fever.universe, [37.5, 39, 40.5])
fever['Muy alta'] = fuzz.trapmf(fever.universe, [39.5, 41, 43, 43])

symptoms['Leves'] = fuzz.trimf(symptoms.universe, [0, 2, 4])
symptoms['Moderados'] = fuzz.trimf(symptoms.universe, [3, 5, 7])
symptoms['Graves'] = fuzz.trapmf(symptoms.universe, [6, 8, 10, 10])

platelets['Bajo'] = fuzz.trapmf(platelets.universe, [50, 50, 100, 150])
platelets['Normal'] = fuzz.trimf(platelets.universe, [100, 200, 300])
platelets['Alto'] = fuzz.trapmf(platelets.universe, [250, 350, 450, 450])

dengue_risk['Bajo'] = fuzz.trimf(dengue_risk.universe, [0, 20, 40])
dengue_risk['Moderado'] = fuzz.trimf(dengue_risk.universe, [30, 50, 70])
dengue_risk['Alto'] = fuzz.trimf(dengue_risk.universe, [60, 80, 100])

# Todas las combinaciones de reglas posibles
rule1 = ctrl.Rule(fever['Normal'] & symptoms['Leves'] & platelets['Bajo'], dengue_risk['Bajo'])
rule2 = ctrl.Rule(fever['Normal'] & symptoms['Leves'] & platelets['Normal'], dengue_risk['Bajo'])
rule3 = ctrl.Rule(fever['Normal'] & symptoms['Leves'] & platelets['Alto'], dengue_risk['Bajo'])

rule4 = ctrl.Rule(fever['Normal'] & symptoms['Moderados'] & platelets['Bajo'], dengue_risk['Moderado'])
rule5 = ctrl.Rule(fever['Normal'] & symptoms['Moderados'] & platelets['Normal'], dengue_risk['Moderado'])
rule6 = ctrl.Rule(fever['Normal'] & symptoms['Moderados'] & platelets['Alto'], dengue_risk['Bajo'])

rule7 = ctrl.Rule(fever['Normal'] & symptoms['Graves'] & platelets['Bajo'], dengue_risk['Alto'])
rule8 = ctrl.Rule(fever['Normal'] & symptoms['Graves'] & platelets['Normal'], dengue_risk['Moderado'])
rule9 = ctrl.Rule(fever['Normal'] & symptoms['Graves'] & platelets['Alto'], dengue_risk['Moderado'])

rule10 = ctrl.Rule(fever['Alta'] & symptoms['Leves'] & platelets['Bajo'], dengue_risk['Moderado'])
rule11 = ctrl.Rule(fever['Alta'] & symptoms['Leves'] & platelets['Normal'], dengue_risk['Bajo'])
rule12 = ctrl.Rule(fever['Alta'] & symptoms['Leves'] & platelets['Alto'], dengue_risk['Bajo'])

rule13 = ctrl.Rule(fever['Alta'] & symptoms['Moderados'] & platelets['Bajo'], dengue_risk['Alto'])
rule14 = ctrl.Rule(fever['Alta'] & symptoms['Moderados'] & platelets['Normal'], dengue_risk['Moderado'])
rule15 = ctrl.Rule(fever['Alta'] & symptoms['Moderados'] & platelets['Alto'], dengue_risk['Moderado'])

rule16 = ctrl.Rule(fever['Alta'] & symptoms['Graves'] & platelets['Bajo'], dengue_risk['Alto'])
rule17 = ctrl.Rule(fever['Alta'] & symptoms['Graves'] & platelets['Normal'], dengue_risk['Moderado'])
rule18 = ctrl.Rule(fever['Alta'] & symptoms['Graves'] & platelets['Alto'], dengue_risk['Moderado'])

rule19 = ctrl.Rule(fever['Muy alta'] & symptoms['Leves'] & platelets['Bajo'], dengue_risk['Alto'])
rule20 = ctrl.Rule(fever['Muy alta'] & symptoms['Leves'] & platelets['Normal'], dengue_risk['Moderado'])
rule21 = ctrl.Rule(fever['Muy alta'] & symptoms['Leves'] & platelets['Alto'], dengue_risk['Moderado'])

rule22 = ctrl.Rule(fever['Muy alta'] & symptoms['Moderados'] & platelets['Bajo'], dengue_risk['Alto'])
rule23 = ctrl.Rule(fever['Muy alta'] & symptoms['Moderados'] & platelets['Normal'], dengue_risk['Alto'])
rule24 = ctrl.Rule(fever['Muy alta'] & symptoms['Moderados'] & platelets['Alto'], dengue_risk['Moderado'])

rule25 = ctrl.Rule(fever['Muy alta'] & symptoms['Graves'] & platelets['Bajo'], dengue_risk['Alto'])
rule26 = ctrl.Rule(fever['Muy alta'] & symptoms['Graves'] & platelets['Normal'], dengue_risk['Alto'])
rule27 = ctrl.Rule(fever['Muy alta'] & symptoms['Graves'] & platelets['Alto'], dengue_risk['Alto'])


# Crear un sistema de control
dengue_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, 
                                  rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, 
                                  rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, 
                                  rule26, rule27])

@app.route('/dengue_risk', methods=['POST'])
def dengue_risk():
    data = request.get_json()
    # Process data and calculate risk
    risk = calculate_dengue_risk(data)
    return jsonify({'dengue_risk': risk, 'data': data})

def calculate_dengue_risk(data):
    # Create a simulation for the fuzzy control system
    dengue_simulation = ctrl.ControlSystemSimulation(dengue_ctrl)
    
    # Assign the input values
    dengue_simulation.input['fever'] = data['fever']
    dengue_simulation.input['symptoms'] = data['symptoms']
    dengue_simulation.input['platelets'] = data['platelets']
    
    # Compute the output
    dengue_simulation.compute()
    
    # Return the risk value
    return dengue_simulation.output['dengue_risk']

if __name__ == '__main__':
    app.run(debug=True)