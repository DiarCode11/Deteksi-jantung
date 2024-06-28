import pandas as pd 
import joblib

# Function to load the model and predict probabilities
def load_model_and_predict_proba(model_path, custom_data):
    # Load the model from the file
    loaded_model = joblib.load(model_path)
    
    # Predict probabilities on the custom data
    probabilities = loaded_model.predict_proba(custom_data)
    return probabilities

# Custom data for testing
custom_data = pd.DataFrame({
    'age': [55, 60],
    'sex': [1, 0],
    'cp': [2, 2],
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
model_path = 'SVM.pkl'  # Change this to the model you want to test

# Load the model and predict probabilities
probabilities = load_model_and_predict_proba(model_path, custom_data)

# Print the probabilities
print("Custom data probabilities:")
for i, proba in enumerate(probabilities):
    print(f"Data {i+1}: Class 0: {proba[0]:.4f}, Class 1: {proba[1]:.4f}")