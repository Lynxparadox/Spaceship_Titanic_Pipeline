import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import sys

st.write("Python version:", sys.version)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "pipeline_model.pkl"

@st.cache_resource
def load_pipeline():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}")
        st.stop()

def main():
    st.title("ASG 05 MD - Angelyn Nathasya M - Spaceship Titanic")

    model = load_pipeline()

    home_planet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
    cryo_sleep = st.selectbox("CryoSleep", [False, True])
    cabin_deck = st.selectbox("CabinDeck", ["A","B","C","D","E","F","G","T","Unknown"])
    cabin_num = st.number_input("CabinNum", 0, 2000, 100)
    cabin_side = st.selectbox("CabinSide", ["P","S","Unknown"])
    destination = st.selectbox("Destination", ["TRAPPIST-1e","PSO J318.5-22","55 Cancri e"])
    age = st.number_input("Age", 0.0, 100.0, 27.0)
    vip = st.selectbox("VIP", [False, True])
    room_service = st.number_input("RoomService", 0.0, 10000.0, 0.0)
    food_court = st.number_input("FoodCourt", 0.0, 10000.0, 0.0)
    shopping_mall = st.number_input("ShoppingMall", 0.0, 10000.0, 0.0)
    spa = st.number_input("Spa", 0.0, 10000.0, 0.0)
    vr_deck = st.number_input("VRDeck", 0.0, 10000.0, 0.0)


    if st.button("Predict"):
        
        data = pd.DataFrame([{
            "HomePlanet": home_planet,
            "CryoSleep": str(cryo_sleep),
            "CabinDeck": cabin_deck,
            "CabinNum": cabin_num,
            "CabinSide": cabin_side,
            "Destination": destination,
            "Age": age,
            "VIP": str(vip),
            "RoomService": room_service,
            "FoodCourt": food_court,
            "ShoppingMall": shopping_mall,
            "Spa": spa,
            "VRDeck": vr_deck
        }])

        prediction = model.predict(data)[0]
        
        if prediction == 1:
            st.success("Passenger will be transported!")
        else:
            st.success("Passenger will not be transported!")
        
if __name__ == "__main__":
    main()