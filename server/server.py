# server.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS  
import pandas as pd
import os

# Load the trained model and threshold
model_path = os.path.join(os.path.dirname(__file__), 'water_potability_model.pkl')
model = joblib.load(model_path)
threshold_path = os.path.join(os.path.dirname(__file__), 'best_threshold.npy')
threshold = np.load(threshold_path)

app = Flask(__name__)

CORS(app, resources={r"/predict": {"origins": ["http://localhost:3000"]}});


@app.route('/', methods=['GET'])
def home():
    # return "Water Potability Prediction Server is running!"
    return jsonify({
        'Sucessfully Connected to Jernih Server'
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Read coordinates (optional, for future database logging)
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    # Expected input: JSON with keys matching your features
    features = [
        data.get('ph'),
        data.get('hardness'),
        data.get('solids'),
        data.get('chloramines'),
        data.get('sulfate'),
        data.get('conductivity'),
        data.get('organic_carbon'),
        data.get('trihalomethanes'),
        data.get('turbidity')
    ]
    feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

    print("DATA RECEIVED:", data)
    print("FEATURES:", features)
    print("ISNA CHECK:\n", pd.DataFrame([features], columns=feature_names).isna())


    features_df = pd.DataFrame([features], columns=feature_names)

    # Predict probability
    prob = model.predict_proba(features_df)[0][1]

    # Apply dynamic threshold
    prediction = int(prob >= threshold)
    # Compute severity
    if prob >= 0.8:
        severity = "Tinggi"
    elif prob >= 0.5:
        severity = "Sedang"
    else:
        severity = "Rendah"

    return jsonify({
        'potability_prediction': prediction,
        'probability': float(prob),
        'threshold_used': float(threshold),
        'severity': severity,
        'latitude': latitude,
        'longitude': longitude
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=5000, host='0.0.0.0')
