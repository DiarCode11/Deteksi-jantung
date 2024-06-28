from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Path to the saved model file
MODEL_PATH = 'SVM.pkl'  # Update with your model path

# Load the model
loaded_model = joblib.load(MODEL_PATH)

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        json_data = request.get_json()
        
        # Convert JSON to pandas DataFrame
        custom_data = pd.DataFrame(json_data)
        
        # Predict probabilities on the custom data
        probabilities = loaded_model.predict_proba(custom_data)
        
        # Prepare response JSON
        response = {
            'predictions': []
        }
        
        # Format predictions
        for i, proba in enumerate(probabilities):
            prediction = {
                'data_id': i + 1,
                'probability_class_0': float(proba[0]),
                'probability_class_1': float(proba[1])
            }
            response['predictions'].append(prediction)
        
        # Return JSON response
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Main driver function
if __name__ == '__main__':
    app.run(debug=True)