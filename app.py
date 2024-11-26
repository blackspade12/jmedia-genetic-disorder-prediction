from flask import Flask, request, send_file, jsonify
from flask_cors import CORS  # Import CORS
from model_predict import predict_genetic_disorder 
import pandas as pd
import json


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/genetic_prediction", methods=["POST", "GET"])
def genetic_predict():
    input_data = request.get_json()
    # Example usage:
    """
    input_data = {
        'White Blood cell count (thousand per microliter)': [0.6529786],
        'Blood cell count (mcL)': [-0.5118449],
        'Patient Age': [-0.714285714],
        'Father\'s age': [0],
        'Mother\'s age': [0],
        'No. of previous abortion': [-0.666666667],
        'Blood test result': [2],
        'Gender': [0],
        'Birth asphyxia': [0],
        'Symptom 5': [1],
        'Heart Rate (rates/min)': [0],
        'Respiratory Rate (breaths/min)': [0],
        'Folic acid details (peri-conceptional)': [1],
        'History of anomalies in previous pregnancies': [1],
        'Autopsy shows birth defect (if applicable)': [0],
        'Assisted conception IVF/ART': [1],
        'Symptom 4': [0],
        'Follow-up': [1],
        'Birth defects': [0],
    }
    """


    predicted_output=predict_genetic_disorder(input_data)

    return jsonify(predicted_output)


if __name__ == '__main__':
    app.run(debug=True)
