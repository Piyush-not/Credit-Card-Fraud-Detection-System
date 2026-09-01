from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import random


app = Flask(__name__)
@app.route("/demo-transaction", methods=["GET"])
def demo_transaction():

    transaction_type = request.args.get(
        "type",
        "legit"
    ).lower()

    data_path = os.path.join(
        "data",
        "creditcard.csv"
    )

    if not os.path.exists(data_path):

        return jsonify({
            "error": "Dataset not found."
        }), 404


    df = pd.read_csv(data_path)


    if transaction_type == "fraud":

        samples = df[
            df["Class"] == 1
        ]

    else:

        samples = df[
            df["Class"] == 0
        ]


    if samples.empty:

        return jsonify({
            "error":
                f"No {transaction_type} transactions found."
        }), 404


    row = samples.sample(
        n=1
    ).iloc[0]


    transaction = {

        "Time":
            float(row["Time"]),

        "Amount":
            float(row["Amount"])

    }


    for i in range(1, 29):

        column = f"V{i}"

        transaction[column] = \
            float(row[column])


    return jsonify({

        "type":
            "fraud"
            if int(row["Class"]) == 1
            else "legit",

        "transaction":
            transaction

    })
# Allow frontend requests during development
CORS(app)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

RF_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest.joblib"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("Random Forest model loaded successfully.")
    print("Scaler loaded successfully.")

except Exception as e:

    model = None
    scaler = None

    print("ERROR loading model:")
    print(e)


# ============================================================
# EXPECTED FEATURES
# ============================================================

FEATURES = [
    "Time",

    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",

    "Amount"
]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "FraudGuard API",
        "status": "running"
    })


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model": "random_forest",
        "model_loaded": model is not None
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None or scaler is None:

        return jsonify({
            "error": "Model or scaler could not be loaded."
        }), 500


    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "No JSON data received."
            }), 400


        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        threshold = float(
            data.get("threshold", 0.5)
        )

        if threshold < 0 or threshold > 1:

            return jsonify({
                "error": "Threshold must be between 0 and 1."
            }), 400


        # ----------------------------------------------------
        # Check features
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in data
        ]

        if missing_features:

            return jsonify({
                "error": "Missing features.",
                "missing_features": missing_features
            }), 400


        # ----------------------------------------------------
        # Create dataframe
        # ----------------------------------------------------

        row = {
            feature: float(data[feature])
            for feature in FEATURES
        }

        df = pd.DataFrame(
            [row],
            columns=FEATURES
        )


        # ----------------------------------------------------
        # Scale Time and Amount
        #
        # Same preprocessing used by the project.
        # ----------------------------------------------------

        df[["Time", "Amount"]] = scaler.transform(
            df[["Time", "Amount"]]
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        probability = model.predict_proba(df)[0][1]

        is_fraud = probability >= threshold


        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        if probability >= 0.70:

            risk_level = "HIGH"

        elif probability >= 0.40:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "fraud_probability": round(
                float(probability),
                6
            ),

            "fraud_percentage": round(
                float(probability * 100),
                2
            ),

            "is_fraud": bool(is_fraud),

            "risk_level": risk_level,

            "threshold": threshold,

            "model": "Random Forest"

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )