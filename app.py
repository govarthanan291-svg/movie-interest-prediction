import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="🎬 Movie Interest Prediction", layout="centered")

st.title("🎬 Movie Interest Prediction App")
st.write("Decision Tree Model using Movie Interest Dataset")

# Load model
model = joblib.load("movie_decision_tree_model.pkl")

# Load dataset (to get feature names and label encoders)
df = pd.read_csv("movie_decision_tree_model.pkl")

# Separate features (same as training)
X = df.drop("Interest", axis=1)

# Handle categorical variables encoding the same way as in training
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = joblib.load(f"label_encoder_{col}.pkl")  # Load pre-trained label encoders
    label_encoders[col] = le

st.subheader("Enter User Details 👇")

# Collect user input dynamically
user_input = {}

for col in X.columns:
    if df[col].dtype == "object":
        # For categorical columns, use selectbox and encode
        user_input[col] = st.selectbox(
            f"{col}",
            options=df[col].unique()
        )
    else:
        # For numerical columns, use number_input
        user_input[col] = st.number_input(
            f"{col}",
            min_value=float(df[col].min()),
            max_value=float(df[col].max()),
            value=float(df[col].mean())
        )

# Convert user input into DataFrame
input_df = pd.DataFrame([user_input])

# Handle categorical variables by encoding the user input the same way as during training
for col in X.select_dtypes(include=["object"]).columns:
    if col in user_input:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Ensure the input dataframe has the same features as the training data
input_df = input_df.reindex(columns=X.columns, fill_value=0)  # Add missing columns if any

# Make prediction
prediction = model.predict(input_df)

# Show prediction result
st.subheader("Prediction Result 🎬")
if prediction[0] == 1:
    st.write("The user is likely interested in movies! 🎥🍿")
else:
    st.write("The user is likely not interested in movies. 😞")

