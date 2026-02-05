import streamlit as st
import pandas as pd
import joblib
import os

# Check if the model file exists
if not os.path.exists("movie_decision_tree_model.pkl"):
    st.error("Model file not found!")
else:
    # Load the trained model
    model = joblib.load("movie_decision_tree_model.pkl")

# Load label encoders
label_encoders = {}
df = pd.read_csv("movie_interest_data.csv")  # Make sure the dataset file is in the same directory
X = df.drop("Interest", axis=1)

# Load the label encoders for each categorical column
for col in X.select_dtypes(include=["object"]).columns:
    label_encoders[col] = joblib.load(f"label_encoder_{col}.pkl")

# Page configuration
st.set_page_config(page_title="🎬 Movie Interest Prediction", layout="centered")

# Streamlit app title and description
st.title("🎬 Movie Interest Prediction App")
st.write("Enter the details below to predict if a user is interested in movies.")

# Collect user input for features
user_input = {}

for col in X.columns:
    if df[col].dtype == "object":
        user_input[col] = st.selectbox(f"{col}", options=df[col].unique())
    else:
        user_input[col] = st.number_input(f"{col}", min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].mean()))

# Convert the user input into a DataFrame
input_df = pd.DataFrame([user_input])

# Handle categorical features by encoding the input the same way as during training
for col in X.select_dtypes(include=["object"]).columns:
    if col in user_input:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Ensure the input DataFrame has the same features as the training data
input_df = input_df.reindex(columns=X.columns, fill_value=0)  # Add missing columns if any

# Make the prediction
prediction = model.predict(input_df)

# Display the result
st.subheader("Prediction Result 🎬")
if prediction[0] == 1:
    st.write("The user is likely interested in movies! 🎥🍿")
else:
    st.write("The user is likely not interested in movies. 😞")
