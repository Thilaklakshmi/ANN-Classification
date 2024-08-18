import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

## Load the trained model
model = tf.keras.models.load_model("model.h5")

## load the encoder and scaler
with open("encoder.pkl","rb") as file:
    encoder = pickle.load(file)

with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender = pickle.load(file)

with open("scaler.pkl","rb") as file:
    scaler= pickle.load(file)

## Streamlit app
st.title("Customer Chirn Prediction")

## User input
geography = st.selectbox("Geography",encoder.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age",18,92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox("HAs Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])

# Preapre the input data
input_data = pd.DataFrame( {
    "CreditScore" : [credit_score],
    "Geography" :[encoder.transform([[geography]])[0]],
    "Gender" : [label_encoder_gender.transform([gender])[0]] ,
    "Age" : [age],
    "Tenure" : [tenure],
    "Balance" : [balance],
    "NumOfProducts" : [num_of_products],
    "HasCrCard" : [has_cr_card],
    "IsActiveMember" : [is_active_member],
    "EstimatedSalary":[estimated_salary]
}
)
## One hot encod "Geography"
geo_encoded = encoder.transform([[geography]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoded,columns=encoder.get_feature_names_out(["Geography"]))


## Combine one hot encoded columns with input data
input_data = pd.concat([input_data.drop("Geography",axis=1),geo_encoder_df],axis=1)

# Now transform the data
input_scaled = scaler.transform(input_data)

## Predict churn
prediction = model.predict(input_scaled)
prediction_probability = prediction[0][0]

st.write(f"Charn Probability:{prediction_probability:.2f}")

if prediction_probability > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")