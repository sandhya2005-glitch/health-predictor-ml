from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")

def home():

    return render_template(
        "index.html",
        prediction=None
    )

@app.route("/predict", methods=["POST"])

def predict():

    gender = request.form["gender"]

    age = float(request.form["age"])

    height = float(request.form["height"])

    weight = float(request.form["weight"])

    sleep = float(request.form["sleep"])

    quality = float(request.form["quality"])

    activity = float(request.form["activity"])

    stress = float(request.form["stress"])

    heart = float(request.form["heart"])

    steps = float(request.form["steps"])

    # Gender Encoding

    gender_encoded = 1 if gender == "Male" else 0

    # BMI Calculation

    bmi = weight / ((height / 100) ** 2)

    # BMI Category

    if bmi < 25:

        bmi_category = 0

    elif bmi < 30:

        bmi_category = 1

    else:

        bmi_category = 2

    # Prediction Data

    data = np.array([[
        gender_encoded,
        age,
        sleep,
        quality,
        activity,
        stress,
        bmi_category,
        heart,
        steps
    ]])

    # Prediction

    prediction = model.predict(data)[0]

    # Result

    if prediction == 0:

        result = "Healthy Lifestyle"
        score = 92
        color = "#22c55e"

        tip = "Excellent health habits. Maintain your healthy daily routine."

    elif prediction == 1:

        result = "Moderate Health Risk"
        score = 65
        color = "#facc15"

        tip = "Improve sleep quality, exercise, and stress management."

    else:

        result = "High Health Risk"
        score = 38
        color = "#ef4444"

        tip = "High stress detected. Improve lifestyle and consult a doctor."

    return render_template(
        "index.html",

        prediction=result,
        score=score,
        color=color,
        bmi=round(bmi,1),
        tip=tip,

        gender=gender,
        age=age,
        height=height,
        weight=weight,
        sleep=sleep,
        quality=quality,
        activity=activity,
        stress=stress,
        heart=heart,
        steps=steps
    )

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)
