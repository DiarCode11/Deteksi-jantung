# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import io
data = pd.read_csv("heart.csv")

# Display the first few rows of the dataframe
print(data.head())

# Display information about the dataset
print(data.info())

# Identifying categorical and numerical columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# Separating features and target
X = data.drop('target', axis=1)
y = data['target']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create preprocessing and modeling pipeline
def create_pipeline(model):
    return Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Initialize the models
models = {
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Train the models and store their accuracy
accuracy_results = {}
pipelines = {}
for model_name, model in models.items():
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results[model_name] = accuracy
    pipelines[model_name] = pipeline
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Visualize the comparison of model performance
plt.figure(figsize=(10, 6))
plt.bar(accuracy_results.keys(), accuracy_results.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Performance')
plt.ylim(0, 1)
plt.show()

# Function to export selected models
def export_models(selected_models, pipelines):
    for model_name in selected_models:
        if model_name in pipelines:
            joblib.dump(pipelines[model_name], f'{model_name}.pkl')
            print(f"Exported {model_name} model to {model_name}.pkl")
        else:
            print(f"Model {model_name} is not available.")

# Choose models to export
models_to_export = ['SVM', 'Random Forest']  # Change this list to select different models

# Export the selected models
export_models(models_to_export, pipelines)

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

# Load the model and predict probabilities
probabilities = load_model_and_predict_proba(model_path, custom_data)

# Print the probabilities
print("Custom data probabilities:")
for i, proba in enumerate(probabilities):
    print(f"Data {i+1}: Class 0: {proba[0]:.4f}, Class 1: {proba[1]:.4f}")
