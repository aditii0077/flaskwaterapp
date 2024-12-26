from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        # Get input values
        try:
            ph = float(request.form["ph"])
            hardness = float(request.form["hardness"])
            solids = float(request.form["solids"])
            chloramines = float(request.form["chloramines"])
            sulfate = float(request.form["sulfate"])
            conductivity = float(request.form["conductivity"])
            organic_carbon = float(request.form["organic_carbon"])
            trihalomethanes = float(request.form["trihalomethanes"])
            turbidity = float(request.form["turbidity"])
            
            # Scale input
            input_data = scaler.transform([[ph, hardness, solids, chloramines, sulfate,
                                            conductivity, organic_carbon, trihalomethanes, turbidity]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Interpret result
            result = "Potable" if prediction == 1 else "Not Potable"
            return render_template("prediction.html", result=result)
        
        except Exception as e:
            return render_template("prediction.html", result=f"Error: {e}")
    
    return render_template("prediction.html", result=None)
