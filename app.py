from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from datetime import datetime
from src.models.risk_predictor import OSCERiskPredictor
from src.models.bias_detector import BiasDetectionSystem

app = Flask(__name__)
CORS(app)

risk_predictor = OSCERiskPredictor()
bias_detector = BiasDetectionSystem()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': True
    })

@app.route('/api/predict_risk', methods=['POST'])
def predict_risk():
    data = request.get_json()
    student_df = pd.DataFrame([data['student_data']])
    risk_score = float(risk_predictor.predict_risk(student_df)[0])
    return jsonify({
        'risk_score': risk_score,
        'risk_level': 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.7 else 'high'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
