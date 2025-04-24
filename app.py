import numpy as np
import pandas as pd
import streamlit as st
import pickle


from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://3836bhagatsingh:3836bhagat@cluster0.0siayoe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client['ice_cream']
collection = db['Pred']



def load_model():
    with open('./selling_pred.pkl','rb') as f:
        model,scaler = pickle.load(f)
    return model,scaler

def preproccesing(data,scaler):
    df = pd.DataFrame(data)
    df_scaled = scaler.transform(df)
    return df_scaled

def prediction(data,model):
    return model.predict(data)

def main():

    st.header("Ice Cream selling prediction with Temp")
    st.subheader("Please enter temperature")

    temp = st.number_input('Enter the temperature')

    store_data = {"temp":temp}

    if st.button('Predict'):
        model,scaler = load_model()
        data = preproccesing([temp],scaler)
        result = prediction(data,model)
        store_data['output'] = abs(int(result))

        st.success(f"Predicted sold ice cream  is {abs(int(result))}")
        collection.insert_one(store_data)

   


if __name__ == '__main__':
    main()

