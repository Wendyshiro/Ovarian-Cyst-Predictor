from flask import Flask, request, jsonify
from fhir_integration import FHIRIntegration  # Ensure this matches your actual Python file name without the .py extension
 
app = Flask(__name__)
fhir = FHIRIntegration()

@app.route('/fhir/patient', methods=['POST'])
def create_patient():
    data = request.json
    patient = fhir.create_patient_resource(data)
    return jsonify(patient), 200

@app.route('/fhir/observation', methods=['POST'])
def create_observation():
    data = request.json
    patient_data = data.get("patient_data")
    observation_type = data.get("observation_type")
    obs = fhir.create_observation_resource(patient_data, observation_type)
    return jsonify(obs), 200

@app.route('/fhir/condition', methods=['POST'])
def create_condition():
    data = request.json
    patient_data = data.get("patient_data")
    prediction = data.get("prediction_result")
    cond = fhir.create_condition_resource(patient_data, prediction)
    return jsonify(cond), 200

@app.route('/fhir/care-plan', methods=['POST'])
def create_care_plan():
    data = request.json
    patient_data = data.get("patient_data")
    care_template = data.get("care_template")
    care = fhir.create_care_plan_resource(patient_data, care_template)
    return jsonify(care), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
