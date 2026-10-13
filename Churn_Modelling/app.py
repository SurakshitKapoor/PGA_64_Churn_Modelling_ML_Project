

import streamlit as st
import joblib
import pandas as pd

# loading the model in current file!
model = joblib.load("dt_model_pipeline.pkl")
print("Model is loaded successfully!")
print("Model is: ", model)

# collect the inputs from user -> from UI -> user!

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

st.subheader("Numerical Features")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score", min_value=0, max_value=1000, value=650
        )
        age = st.number_input(
            "Age", min_value=18, max_value=120, value=35
        )
        tenure = st.number_input(
            "Tenure (years)", min_value=0, max_value=10, value=5
        )
        balance = st.number_input(
            "Balance", min_value=0.0, value=75000.0
        )
        num_products = st.selectbox(
            "Number of Products", [1, 2, 3, 4]
        )

    with col2:
        has_card = st.selectbox("Has Credit Card?", [1, 0])
        active_member = st.selectbox("Is Active Member?", [1, 0])
        salary = st.number_input(
            "Estimated Salary", min_value=0.0, value=85000.0
        )
        zero_balance = st.selectbox("Zero Balance?", [0, 1])
        balance_per_product = st.number_input(
            "Balance per Product", min_value=0.0, value=37500.0
        )

    st.subheader("Categorical Features")

    col3, col4 = st.columns(2)

    with col3:
        geography = st.selectbox(
            "Geography", ["France", "Germany", "Spain"]
        )

    with col4:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

    submitted = st.form_submit_button(
        "🔍 Predict Churn", use_container_width=True
    )


if submitted:
    input_df = pd.DataFrame([{
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_card,
        "IsActiveMember": active_member,
        "EstimatedSalary": salary,
        "zero_balance": zero_balance,
        "balance_per_product": balance_per_product,
        "Geography": geography,
        "Gender": gender
    }])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn!")
    else:
        st.success("✅ Customer is likely to stay.")