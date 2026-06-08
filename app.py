import pickle
from pathlib import Path
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ml" / "model.pkl"

FEATURE_IMPORTANCE = {
    "glucose": 38,
    "bmi": 21,
    "age": 14,
    "diabetesPedigree": 10,
    "bloodPressure": 7,
    "insulin": 5,
    "skinThickness": 3,
    "pregnancies": 2,
}

CSV_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

FIELD_RANGES = {
    "pregnancies": (0, 17),
    "glucose": (50, 300),
    "bloodPressure": (30, 140),
    "skinThickness": (0, 99),
    "insulin": (0, 846),
    "bmi": (10, 70),
    "diabetesPedigree": (0.05, 2.5),
    "age": (1, 120),
}

OPTIONAL_DEFAULTS = {
    "pregnancies": 0,
    "skinThickness": 20.0,
    "insulin": 79.0,
    "diabetesPedigree": 0.47,
}


def load_model(path):
    if not path.exists():
        return None
    with open(path, "rb") as file:
        return pickle.load(file)


def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def get_contributing_factors(data):
    factors = []
    if data["glucose"] > 140:
        factors.append("High glucose level detected")
    if data["bmi"] > 30:
        factors.append("BMI indicates obesity")
    if data["age"] > 45:
        factors.append("Age is a risk factor (above 45)")
    if data["bloodPressure"] > 90:
        factors.append("Elevated blood pressure")
    if data["insulin"] > 200:
        factors.append("High insulin level")
    if data["diabetesPedigree"] > 0.8:
        factors.append("Strong family history of diabetes")
    return factors or ["No significant individual risk factors detected"]


def get_health_tips(is_diabetic, data):
    if is_diabetic:
        return [
            "Consult a doctor immediately for clinical evaluation.",
            "Reduce sugar and refined carbohydrate intake.",
            "Aim for at least 30 minutes of exercise daily.",
            "Monitor blood glucose levels regularly.",
            "Maintain a healthy weight (target BMI under 25).",
        ]

    tips = ["Maintain your healthy lifestyle."]
    if data["bmi"] > 25:
        tips.append("Work on reducing BMI to below 25.")
    if data["glucose"] > 100:
        tips.append("Watch your sugar intake, glucose is borderline.")
    tips.append("Schedule regular health checkups annually.")
    tips.append("Stay physically active and eat a balanced diet.")
    return tips


def build_input_array(data):
    return np.array([[
        data["pregnancies"],
        data["glucose"],
        data["bloodPressure"],
        data["skinThickness"],
        data["insulin"],
        data["bmi"],
        data["diabetesPedigree"],
        data["age"],
    ]], dtype=float)


def predict(data, model):
    input_array = build_input_array(data)
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0][1]
    diabetes_chance = round(proba * 100, 2)
    confidence = round(abs(proba - 0.5) * 2 * 100, 2)
    if diabetes_chance >= 70:
        risk_level = "High"
    elif diabetes_chance >= 40:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "prediction": bool(prediction == 1),
        "diabetesChance": diabetes_chance,
        "confidence": confidence,
        "riskLevel": risk_level,
        "bmiCategory": get_bmi_category(data["bmi"]),
        "contributingFactors": get_contributing_factors(data),
        "healthTips": get_health_tips(bool(prediction == 1), data),
        "featureImportance": FEATURE_IMPORTANCE,
    }


model = load_model(MODEL_PATH)

st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="centered",
)

st.title("Diabetes Risk Predictor")
st.markdown(
    "Enter your health data below and the model will estimate your diabetes risk."
)

if model is None:
    st.error(f"Model not found at: {MODEL_PATH}\nPlease make sure {MODEL_PATH} exists.")
    st.stop()

with st.form(key="predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=17, value=OPTIONAL_DEFAULTS["pregnancies"], step=1
        )
        glucose = st.number_input("Glucose", min_value=50.0, max_value=300.0, value=120.0)
        bloodPressure = st.number_input("Blood Pressure", min_value=30.0, max_value=140.0, value=70.0)
        skinThickness = st.number_input(
            "Skin Thickness", min_value=0.0, max_value=99.0, value=OPTIONAL_DEFAULTS["skinThickness"]
        )
    with col2:
        insulin = st.number_input("Insulin", min_value=0.0, max_value=846.0, value=OPTIONAL_DEFAULTS["insulin"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=24.0)
        diabetesPedigree = st.number_input(
            "Diabetes Pedigree Function", min_value=0.05, max_value=2.5, value=OPTIONAL_DEFAULTS["diabetesPedigree"], format="%.2f"
        )
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    submitted = st.form_submit_button("Predict")

if submitted:
    user_input = {
        "pregnancies": pregnancies,
        "glucose": glucose,
        "bloodPressure": bloodPressure,
        "skinThickness": skinThickness,
        "insulin": insulin,
        "bmi": bmi,
        "diabetesPedigree": diabetesPedigree,
        "age": age,
    }
    result = predict(user_input, model)

    st.subheader("Prediction Result")
    st.write("**Diabetes risk detected:**", "Yes" if result["prediction"] else "No")
    st.write("**Diabetes chance:**", f"{result['diabetesChance']}%")
    st.write("**Confidence:**", f"{result['confidence']}%")
    st.write("**Risk level:**", result["riskLevel"])
    st.write("**BMI category:**", result["bmiCategory"])

    st.markdown("---")
    st.subheader("Contributing factors")
    for factor in result["contributingFactors"]:
        st.write(f"- {factor}")

    st.subheader("Health tips")
    for tip in result["healthTips"]:
        st.write(f"- {tip}")

    st.subheader("Feature importance")
    st.table(
        sorted(
            [{"Feature": k, "Importance": v} for k, v in result["featureImportance"].items()],
            key=lambda row: row["Importance"],
            reverse=True,
        )
    )

    st.info("This is a model prediction only and not medical advice.")
else:
    st.write("Fill in the values and click Predict to see the result.")
