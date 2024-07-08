import streamlit as st
import numpy as np
import joblib
import pandas as pd

pipeline = joblib.load('california_housing_pipe.pkl')


def get_user_input():
    MedInc = st.number_input("Enter the median income of the block", 0.0, 20.0, 1.0)
    HouseAge = st.number_input("Enter the housing median age", 0, 60, 20)
    AveRooms = st.number_input("Enter the average number of rooms", 0, 150, 5)
    AveBedrms = st.number_input("Enter the average number of bedrooms", 0, 40, 1)
    Population = st.number_input("Enter the population", 0, 40000, 1000)
    AveOccup = st.number_input("Enter the average occupancy", 0, 1300, 3)
    Latitude = st.number_input("Enter the latitude", 35.0, 42.0, 37.0)
    Longitude = st.number_input("Enter the longitude", -130.0, -113.0, -116.0)
    
    
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    return data


def main():
    st.title("California House Price Prediction")
    st.write("Enter the details of the house to predict its price:")

    data = get_user_input()
    
    if st.button("Predict"):
        data = pd.DataFrame(data, index=[0])
        prediction = pipeline.predict(data)
        st.write(f"The predicted house price is ${prediction[0]:,.2f}")
        
if __name__ == "__main__":
    main()