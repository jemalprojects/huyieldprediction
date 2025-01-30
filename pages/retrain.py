from  PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
import io 
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as html
import tensorflow as tf
import time
import math
import data_processing
import prediction
from PIL import Image

from huggingface_hub import snapshot_download
st.subheader("Retrain Model Interface", divider=True)
df = pd.read_csv('bbox_and_commons.csv')
districts = df['district'].tolist()
district_selected = st.multiselect("Select Districts", districts)
# url=f"https://huggingface.co/datasets/abatejemal/2_Data/resolve/main/WeatherData/{district}.csv"
# dataset_paths = [f"2_Data/WeatherData/{district}.csv" for district in district_selected]
dataset_paths = [f"https://huggingface.co/datasets/abatejemal/2_Data/resolve/main/WeatherData/{district}.csv" for district in district_selected]


# dataset_paths = [f"{path1}/{district}.csv" for district in district_selected]
# model_paths = [f"3_Models/weather_models/{district}_lstm_model.h5" for district in district_selected]
# scaler_paths = [f"3_Models/weather_models/{district}_scaler.pkl" for district in district_selected]
# Load dataset if districts are selected

if district_selected:
    for district, dataset_path in zip(district_selected, dataset_paths):
        # st.write(dataset_path)
        data = pd.read_csv(dataset_path)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        numeric_columns = ['GWETPROF', 'GWETTOP', 'GWETROOT', 'CLOUD_AMT', 'TS', 'PS', 'RH2M', 'QV2M', 'PRECTOTCORR', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS2M']
        data = data[numeric_columns].dropna()
        data = data_processing.fill_outliers_with_median(data)
        # if os.path.exists(dataset_path):
            # st.write(dataset_path)
            # data = pd.read_csv(dataset_path)
            # data['date'] = pd.to_datetime(data['date'])
            # data.set_index('date', inplace=True)
            # numeric_columns = ['GWETPROF', 'GWETTOP', 'GWETROOT', 'CLOUD_AMT', 'TS', 'PS', 'RH2M', 'QV2M', 'PRECTOTCORR', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'WS2M']
            # data = data[numeric_columns].dropna()
            # data = data_processing.fill_outliers_with_median(data)
        # else:
            # st.write(f"No dataset found for {district}. Please check the file path.")
    if "processing" not in st.session_state:
        st.session_state.processing = False
    # Create a placeholder for the button
    button_placeholder = st.empty()
    
    # Show the button only if not processing
    if not st.session_state.processing:
        # Show the button with a unique key using a combination of the processing status
        if button_placeholder.button("Retrain", key="run_task_button_visible"):
            st.session_state.processing = True
            button_placeholder.empty()  # Hide the button during task
            # with st.spinner("Please Wait, Training Model..."):
            with st.status("Please Wait, Training Model...", expanded=True) as status:
                # executor.submit(model_training.retrain_model_function(district_selected, dataset_paths))
                model_training.retrain_model_function(district_selected, dataset_paths)
                st.session_state.processing = False  # Reset the processing flag
                button_placeholder.button("Retrain", key="run_task_button_complete")
                status.update(label="Training complete!", state="complete", expanded=True)
