# Disease Prediction System

A machine learning-based web application that predicts possible diseases based on user-selected symptoms. The system uses a Random Forest classifier trained on medical symptom data to provide probability-based predictions.

## Features

- **Symptom Selection**: Interactive web interface to select multiple symptoms from a comprehensive list.
- **Real-time Search**: Search functionality to quickly find symptoms.
- **Probability Predictions**: Displays top 3 disease predictions with probability scores.
- **Responsive Design**: Mobile-friendly interface using Bootstrap.
- **REST API**: Backend API endpoint for predictions.
- **Logging**: Comprehensive logging for debugging and monitoring.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. Clone or download the project files to your local machine.

2. Navigate to the project directory:
   ```
   cd disease_prediction
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Ensure the following files are present in the project directory:
   - `disease_model.pkl` (trained model)
   - `symptoms.pkl` (symptom list)
   - `dataset/Training.csv` and `dataset/Testing.csv` (if retraining is needed)

## Usage

### Running the Application

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`

3. Select symptoms from the list (use the search box to filter).

4. Click "Predict" to get disease predictions.

5. Review the results in the modal popup showing the top 3 diseases with probabilities.

### API Usage

The application provides a REST API endpoint for predictions:

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "symptoms": ["symptom1", "symptom2", "symptom3"]
}
```

**Response:**
```json
{
  "predictions": [
    ["Disease Name 1", 0.85],
    ["Disease Name 2", 0.12],
    ["Disease Name 3", 0.03]
  ]
}
```

## Dataset

The model is trained on a medical symptom dataset containing:
- **Training Data**: `dataset/Training.csv`
- **Testing Data**: `dataset/Testing.csv`

Each dataset includes various symptoms as features and the corresponding disease prognosis as the target variable.

## Model

- **Algorithm**: Random Forest Classifier
- **Training**: Performed using scikit-learn with 100 estimators
- **Features**: Binary symptom presence (0 or 1)
- **Output**: Probability scores for each possible disease
- **Accuracy**: Evaluated on test data (see training script output)

To retrain the model, uncomment the `train_and_test_model()` call in `train_model.py` and run:
```
python train_model.py
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and evaluation
- **flask**: Web framework for the application
- **joblib**: Model serialization and loading

## Project Structure

```
disease_prediction/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── disease_model.pkl      # Trained model file
├── symptoms.pkl           # Symptom list file
├── requirements.txt       # Python dependencies
├── TODO.md                # Pending tasks
├── templates/
│   └── index.html         # Web interface
└── dataset/
    ├── Training.csv       # Training dataset
    └── Testing.csv        # Testing dataset
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## TODO

- Remove extra void spaces in HTML
- Add disclaimer in prediction modal that it cannot be accurate, just a prediction
- Add styling to the prediction card

## Disclaimer

This application is for educational and informational purposes only. The predictions are based on machine learning models and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any medical concerns.

## License

This project is open-source and available under the MIT License.
