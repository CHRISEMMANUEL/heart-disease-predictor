from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Load model and columns
model = joblib.load("heart_disease_model.pkl")
model_columns = joblib.load("model_columns.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        # Align user input with training columns
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
    except Exception as e:
        result = f"Error: {e}"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
