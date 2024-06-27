import pandas as pd
import joblib

# Function to load the model and make predictions
def load_model_and_predict(model_path, custom_data):
    # Load the model from the file
    loaded_model = joblib.load(model_path)
    
    # Make predictions on the custom data
    predictions = loaded_model.predict(custom_data)
    probs = loaded_model.predict_proba(custom_data)
    return predictions, probs

# Custom data for testing
custom_data = pd.DataFrame({
    'age': [55, 60],
    'sex': [1, 0],
    'cp': [0, 2],
    'trestbps': [130, 140],
    'chol': [250, 230],
    'fbs': [0, 1],
    'restecg': [1, 0],
    'thalach': [150, 160],
    'exang': [0, 1],
    'oldpeak': [1.5, 2.3],
    'slope': [2, 1],
    'ca': [0, 1],
    'thal': [2, 3]
})

# Path to the saved model file
model_path = 'SVM.pkl'

# Load the model and make predictions
predictions, probs = load_model_and_predict(model_path, custom_data)

# Print the predictions
print(f"Custom data predictions: {predictions} {probs}")
