
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained regression model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit App
st.title('Estimated Salary Prediction')

st.write('Enter the customer details below to predict the estimated salary.')


# User Inputs

credit_score = st.number_input(
    'Credit Score',
    min_value=300,
    max_value=850,
    value=650
)

geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.slider(
    'Age',
    min_value=18,
    max_value=92,
    value=35
)

tenure = st.slider(
    'Tenure',
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    'Balance',
    min_value=0.0,
    value=50000.0
)

num_of_products = st.slider(
    'Number of Products',
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    'Has Credit Card',
    [0, 1]
)

is_active_member = st.selectbox(
    'Is Active Member',
    [0, 1]
)

exited = st.selectbox(
    'Exited',
    [0, 1]
)


# Prepare Input Data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})


# One-hot encode Geography

geo_encoded = onehot_encoder_geo.transform(
    [[geography]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)


# Combine input data with Geography columns

input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)


# Scale the input data

input_data_scaled = scaler.transform(input_data)


# Prediction Button

if st.button('Predict Salary'):

    prediction = model.predict(input_data_scaled, verbose=0)

    predicted_salary = prediction[0][0]

    st.success(
        f'Predicted Estimated Salary: ${predicted_salary:,.2f}'
    )
