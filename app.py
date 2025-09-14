from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and symptoms
try:
    model = joblib.load('disease_model.pkl')
    symptoms = joblib.load('symptoms.pkl')
    logger.info("Model and symptoms loaded successfully")
except Exception as e:
    logger.error(f"Error loading model files: {str(e)}")
    # Create dummy data for testing if files aren't available
    symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering']
    model = None

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
            
        data = request.get_json()
        selected_symptoms = data.get('symptoms', [])
        logger.info(f"Received symptoms: {selected_symptoms}")
        
        # Create symptom vector
        symptom_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
        logger.info(f"Symptom vector: {symptom_vector}")
        
        # Get probabilities for all diseases
        probabilities = model.predict_proba([symptom_vector])[0]
        disease_names = model.classes_
        
        # Create list of (disease, probability) pairs
        disease_probs = list(zip(disease_names, probabilities))
        
        # Sort by probability (descending) and get top 3
        disease_probs.sort(key=lambda x: x[1], reverse=True)
        top_predictions = disease_probs[:3]
        
        logger.info(f"Top predictions: {top_predictions}")
        
        # Convert numpy types to Python native types for JSON serialization
        predictions_serializable = [
            [str(disease), float(probability)] 
            for disease, probability in top_predictions
        ]
        
        return jsonify({'predictions': predictions_serializable})
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)