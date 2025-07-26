from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model/student_score_model.pkl")
encoders = joblib.load("model/encoders.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form
    print(form_data)

    try:
        gender = form_data["gender"]
        race = form_data["race_ethnicity"]
        parental_education = form_data["parental_education"]
        lunch = form_data["lunch"]
        test_prep = form_data["test_preparation"]
        math = float(form_data["math_score"])
        reading = float(form_data["reading_score"])
        writing = float(form_data["writing_score"])

        # Encode categorical inputs using correct encoder keys
        gender_enc = encoders['gender'].transform([gender])[0]
        race_enc = encoders['race/ethnicity'].transform([race])[0]
        parental_enc = encoders['parental_level_of_education'].transform([parental_education])[0]
        lunch_enc = encoders['lunch'].transform([lunch])[0]
        prep_enc = encoders['test_preparation_course'].transform([test_prep])[0]

        input_data = np.array([[gender_enc, race_enc, parental_enc, lunch_enc, prep_enc, math, reading, writing]])
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        return render_template("index.html", prediction=prediction)

    except KeyError as e:
        import traceback
        traceback.print_exc()
        return f"❌ KeyError: {e}<br><br>Form: {form_data}<br><br>Encoders: {list(encoders.keys())}", 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return f"❌ Server error: {e}<br><br>Form: {form_data}", 500

if __name__ == "__main__":
    app.run(debug=True)
