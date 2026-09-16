import pickle
import joblib
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="ASD Prediction",
    page_icon="🧠",
)

st.title("Autism Spectrum Disorder Prediction")
st.caption("Machine-learning demonstration using the trained classification pipeline.")


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_model.pkl"

try:
    model = joblib.load(MODEL_PATH)

except FileNotFoundError:
    st.error("Model file not found. Check that models/final_model.pkl exists.")
    st.stop()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.exception(e)
    st.stop()


with st.form("prediction_form"):

    age = st.number_input(
        "Age",
        min_value=2.0,
        max_value=90.0,
        value=25.0,
        step=0.1,
    )

    gender = st.selectbox(
        "Gender",
        ["m", "f"],
    )

    ethnicity = st.selectbox(
        "Ethnicity",
        [
            "Others",
            "Asian",
            "Black",
            "Hispanic",
            "Latino",
            "Middle Eastern ",
            "Pasifika",
            "South Asian",
            "Turkish",
            "White-European",
        ],
    )

    jaundice = st.selectbox(
        "Jaundice at birth",
        ["no", "yes"],
    )

    austim = st.selectbox(
        "Family history of autism",
        ["no", "yes"],
    )

    country = st.selectbox(
        "Country of residence",
        [
            "Afghanistan", "Angola", "Argentina", "Armenia", "Aruba",
            "Australia", "Austria", "Azerbaijan", "Bahamas", "Bangladesh",
            "Belgium", "Bolivia", "Brazil", "Burundi", "Canada", "China",
            "Cyprus", "Czech Republic", "Egypt", "Ethiopia", "France",
            "Germany", "Hong Kong", "Iceland", "India", "Iran", "Iraq",
            "Ireland", "Italy", "Japan", "Jordan", "Kazakhstan", "Malaysia",
            "Mexico", "Netherlands", "New Zealand", "Nicaragua", "Niger",
            "Oman", "Pakistan", "Romania", "Russia", "Saudi Arabia", "Samoa",
            "Serbia", "Sierra Leone", "South Africa", "Spain", "Sri Lanka",
            "Sweden", "Tonga", "Ukraine", "United Arab Emirates",
            "United Kingdom", "United States", "Vietnam",
        ],
    )

    relation = st.selectbox(
        "Relation",
        ["Self", "Parent", "Relative", "Health care professional", "Others", "?"],
    )

    st.write("### Screening scores")

    scores = {}
    for i in range(1, 11):
        scores[f"A{i}_Score"] = st.selectbox(
            f"A{i}",
            [0, 1],
            key=f"a{i}",
        )

    result = st.number_input(
        "Screening result score",
        value=5.0,
        step=0.1,
    )

    submitted = st.form_submit_button("Predict")



if submitted:

    input_data = pd.DataFrame(
        {
            "A1_Score": [scores["A1_Score"]],
            "A2_Score": [scores["A2_Score"]],
            "A3_Score": [scores["A3_Score"]],
            "A4_Score": [scores["A4_Score"]],
            "A5_Score": [scores["A5_Score"]],
            "A6_Score": [scores["A6_Score"]],
            "A7_Score": [scores["A7_Score"]],
            "A8_Score": [scores["A8_Score"]],
            "A9_Score": [scores["A9_Score"]],
            "A10_Score": [scores["A10_Score"]],
            "age": [age],
            "gender": [gender],
            "ethnicity": [ethnicity],
            "jaundice": [jaundice],
            "austim": [austim],
            "contry_of_res": [country],
            "result": [result],
            "relation": [relation],
        }
    )

    prediction = int(model.predict(input_data)[0])

    st.divider()

    if prediction == 1:
        st.error("Model prediction: ASD")
    else:
        st.success("Model prediction: No ASD")


