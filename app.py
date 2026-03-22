import streamlit as st
import joblib
import numpy as np
model = joblib.load("house_prices_model.pkl")
st.title("House price prediction App")
st.write("Enter house details to predict price")

Area=st.number_input("Area(in sq ft)",min_value=500,max_value=5000,step=50)
Bedrooms=st.number_input("Number of bedrooms",min_value=1,max_value=10,step=1)
Age=st.number_input("Age of House",min_value=0,max_value=50,step=1)

if st.button("Predict Price"):
    input_data=np.array([[Area,Bedrooms,Age]])
    prediction=model.predict(input_data)

    st.success(f"Estimated House Price:{prediction[0]:2f}Lakhs")
st.subheader("Training Dataset")
