import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import cloudpickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = Flask(__name__)

# Load Preprocessor
with open("models/laptop_preprocessor.pkl", "rb") as f:
    preprocessor = cloudpickle.load(f)

# Load XGBOOSt model
with open("models/xgb_model.pkl", "rb") as f:
    model = cloudpickle.load(f)

# Load Random Forest model
with open("models/rf_model.pkl", "rb") as f:
    rf_model = cloudpickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input
    user_input = {
        "Company" : (request.form["Company"]),
        "TypeName": (request.form["TypeName"]),
        "Cpu": (request.form["Cpu"]),
        "Gpu": (request.form["Gpu"]),
        "Ram": (request.form["Ram"]),
        "Memory": (request.form["Memory"]),
        "OpSys": (request.form["OpSys"]),
        "ScreenResolution": (request.form["ScreenResolution"]),
        "Inches": (request.form["Inches"]),
        "Weight": request.form["Weight"]
    }

    input_df = pd.DataFrame([user_input])

    # Applying Preprocessing pipline
    preprocess_input = preprocessor.transform(input_df)

    # Apply trained model and reverse log transformation on price
    xgb_prediction = np.expm1(model.predict(preprocess_input)[0])
    rf_prediction = np.expm1(rf_model.predict(preprocess_input)[0])

    return render_template(
        "result.html",
        xgb_prediction=round(xgb_prediction, 2),
        rf_prediction=round(rf_prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)