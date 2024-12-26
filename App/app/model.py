import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# Path to dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "../data/water_potability.csv")

def train_model():
    # Load the dataset
    try:
        data = pd.read_csv(DATASET_PATH)
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        raise Exception(f"Dataset not found at {DATASET_PATH}. Please ensure the file is in the correct location.")
    
    # Handle missing values
    data = data.dropna()
    
    # Features and target
    X = data.drop("Potability", axis=1)
    y = data["Potability"]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.2f}")
    
    # Save the model and scaler
    joblib.dump(model, os.path.join(os.path.dirname(__file__), "model.pkl"))
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), "scaler.pkl"))
    print("Model and scaler saved successfully!")

# Run this script to train the model
if __name__ == "__main__":
    train_model()
