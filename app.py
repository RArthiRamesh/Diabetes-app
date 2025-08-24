# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("diabetes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get visible inputs from the form
    glucose = float(request.form["Glucose"])
    bloodpressure = float(request.form["BloodPressure"])
    insulin = float(request.form["Insulin"])
    bmi = float(request.form["BMI"])
    age = float(request.form["Age"])
    
    # Hidden/default values
    pregnancies = 0
    skin_thickness = 20
    diabetes_pedigree = 0.5

    # Create features array in the right order
    features = [pregnancies, glucose, bloodpressure, skin_thickness,
                insulin, bmi, diabetes_pedigree, age]

    prediction = model.predict([features])[0]
    output = "Diabetic" if prediction == 1 else "Not Diabetic"

    return render_template("index.html", prediction_text=f"Prediction: {output}")

if __name__ == "__main__":
    app.run(debug=True)
