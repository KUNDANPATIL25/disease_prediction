import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

def train_and_test_model():
    # Load data
    train_data = pd.read_csv('dataset/Training.csv')
    test_data = pd.read_csv('dataset/Testing.csv')

    # Drop unnamed columns if any
    train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
    test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]

    # Separate features and target
    X_train = train_data.drop('prognosis', axis=1)
    y_train = train_data['prognosis']
    X_test = test_data.drop('prognosis', axis=1)
    y_test = test_data['prognosis']

    # Handle missing values (replace NaN with 0)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, 'disease_model.pkl')

    # Save symptom list 
    symptoms = list(X_train.columns)
    joblib.dump(symptoms, 'symptoms.pkl')
    
    print(f"Model trained with {len(symptoms)} symptoms and {len(model.classes_)} diseases")
    print("Model saved as 'disease_model.pkl'")
    print("Symptoms list saved as 'symptoms.pkl'")

def test_prediction_function():
    # Load the saved model and symptoms
    model = joblib.load('disease_model.pkl')
    symptoms = joblib.load('symptoms.pkl')
    
    # Test with some sample symptoms
    test_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions']
    
    print(f"Testing with symptoms: {test_symptoms}")
    
    # Create symptom vector
    symptom_vector = [1 if symptom in test_symptoms else 0 for symptom in symptoms]
    
    # Get probabilities
    probabilities = model.predict_proba([symptom_vector])[0]
    disease_names = model.classes_
    
    # Get top 3 predictions
    disease_probs = list(zip(disease_names, probabilities))
    disease_probs.sort(key=lambda x: x[1], reverse=True)
    top_predictions = disease_probs[:3]
    
    print("\nTop 3 Predictions:")
    for i, (disease, prob) in enumerate(top_predictions, 1):
        print(f"{i}. {disease}: {prob:.2%}")

if __name__ == '__main__':
    print("=== Training and Testing Disease Prediction Model ===\n")

    # Train the model (uncomment if you want to retrain)
    train_and_test_model()

    print("\n=== Testing Prediction Function ===\n")
    test_prediction_function()
