from  PIL import Image
from base64 import b64encode
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
import data_processing
import io 
import joblib
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as html
import tensorflow as tf
import time


time_steps = 365
def build_model(input_shape):
    """
    Builds and compiles an LSTM model.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(input_shape[1])  # Predicting all features
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
    return model


import joblib
from huggingface_hub import HfApi, upload_file, login
import io

def upload_model_to_huggingface(model, model_filename, private=False):
    # Serialize the trained model to a BytesIO object (in-memory storage)
    model_io = io.BytesIO()
    joblib.dump(model, model_io)
    model_io.seek(0)  # Reset the pointer to the beginning of the BytesIO object
    token = st.secrets["huggingface_TOKEN"]
    # Log in to Hugging Face
    login(token=token)
    repo_name = "3_Models"
    username = "abatejemal"

    # Create a repository on Hugging Face Hub (if it doesn't exist)
    # api = HfApi()
    # api.create_repo(repo_name, private=private)  # Set private=True if you want a private repo

    # Upload the model directly to the Hugging Face repository
    upload_file(
        path_or_fileobj=model_io,  # Pass the in-memory model object
        path_in_repo=model_filename,  # Path where the file will be stored in the repo
        repo_id=f"{username}/{repo_name}",  # Replace 'your_username' with your Hugging Face username
        repo_type="model",  # Specify that it's a model
    )

    # print(f"Model uploaded successfully to https://huggingface.co/{username}/{repo_name}")



# upload_model_to_huggingface(regressor_rf, repo_name, token, username)


def retrain_model_function(district_selected, dataset_paths):
    # Main training logic
    total_districts = len(district_selected)
    i = 0
    district_progress = st.progress(0)
    for district, dataset_path in zip(district_selected, dataset_paths):
        i += 1
        progress_desc = f"Processing district ({i}/{total_districts})"
        district_progress.progress(i / total_districts, text=progress_desc)
        data_scaled, scaler, data_original = data_processing.preprocess_data(dataset_path)
        X, y = data_processing.prepare_data(data_scaled, time_steps)

        # Train-test split
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train the model
        model = build_model((time_steps, X.shape[2]))
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        total_epoch = 10
        # progress_bar = st.progress(perc, text=f"{district} ({int(perc)}%)")
        progress_bar = st.progress(0, text=f"{district} (0%)")
        def on_epoch_end(epoch, logs):
            percentage = ((epoch + 1) / total_epoch) * 100
            progress_bar.progress((epoch + 1) / total_epoch, text=f"{district} ({int(percentage)}%)")

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=total_epoch,
            batch_size=32,
            callbacks=[early_stopping, tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)],
            verbose=0
        )

        # Save the model and scaler locally
        os.makedirs("3_Models/weather_models", exist_ok=True)
        model_save_path = f"3_Models/weather_models/{district}_lstm_model.h5"
        scaler_save_path = f"3_Models/weather_models/{district}_scaler.pkl"
        model.save(model_save_path)
        joblib.dump(scaler, scaler_save_path)
        # Function to upload a file to GitHub
        
        # model_repo_path = f"models/{district}_lstm_model.h5"
        model_repo_path = model_save_path
        # scaler_repo_path = f"models/{district}_scaler.pkl"
        scaler_repo_path = scaler_save_path
        model_filename=f"weather_models/{district}_lstm_model.h5"
        upload_model_to_huggingface(model, model_filename)
        scaler_filename=f"weather_models/{district}_scaler.pkl"
        upload_model_to_huggingface(scaler, scaler_filename)
        # upload_to_github(model_save_path, model_repo_path, commit_message_template.format(file_name="model", district=district))
        # upload_to_github(scaler_save_path, scaler_repo_path, commit_message_template.format(file_name="scaler", district=district))

        progress_bar.empty()
        
        # st.write(f"Model and scaler have been retrained and saved for {district}")
    district_progress.empty()
    st.success("Training Completed!")
