from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd

from huggingface_hub import snapshot_download

def predict_crop_yield1(df, encoded_final, ohe_loaded):
    # Load the model directly from Hugging Face (without downloading manually)
    # model_rf = joblib.load(snapshot_download("abatejemal/crop_model", filename="cereals_rf.pkl"))
    model_dir = snapshot_download("abatejemal/crop_model")
    model_rf = joblib.load(f'{model_dir}/cereals_rf.pkl')
    
    # Make predictions
    predictions = model_rf.predict(df)

    # Decode 'crop' and 'season' columns back to original values
    decoded = ohe_loaded.inverse_transform(encoded_final)
    decoded1 = pd.DataFrame(decoded)
    decoded1.columns = ['crop', 'season']
    
    # Create the final DataFrame with relevant features and predictions
    final_result = pd.concat([df[['area(sq.m)', 
                   'GWETPROF', 'GWETTOP', 'GWETROOT', 'CLOUD_AMT', 
                   'TS', 'PS', 'RH2M', 'QV2M', 'PRECTOTCORR', 'T2M_MAX', 
                   'T2M_MIN', 'T2M_RANGE', 'WS2M', 'elevation', 'slope', 'soc', 'soilph',
                               ]], decoded1], axis=1)
    
    # Add predicted values to the DataFrame
    final_result['Predicted'] = predictions

    # Return the DataFrame sorted by predictions
    return final_result.sort_values(by=['Predicted'], ascending=False)



# def predict_crop_yield(df, encoded_final, ohe_loaded):
    # # Load pre-trained model
    # model_rf = joblib.load('3_Models/yield_models/cereals_rf.pkl')
    # # Make predictions
    # predictions = model_rf.predict(df)

    # # Decode 'crop' and 'season' columns back to original values
    # decoded = ohe_loaded.inverse_transform(encoded_final)
    # decoded1=pd.DataFrame(decoded)
    # decoded1.columns=['crop', 'season']
    # final_result = pd.concat([df[['area(sq.m)', 
                   # 'GWETPROF', 'GWETTOP', 'GWETROOT', 'CLOUD_AMT', 
                   # 'TS', 'PS', 'RH2M', 'QV2M', 'PRECTOTCORR', 'T2M_MAX', 
                   # 'T2M_MIN', 'T2M_RANGE', 'WS2M', 'elevation', 'slope', 'soc', 'soilph',
                               # ]], decoded1], axis=1)
    
    # # Add predicted values to DataFrame
    # final_result['Predicted'] = predictions

    # # Return the DataFrame sorted by predictions
    # return final_result.sort_values(by=['Predicted'], ascending=False)
    
# def predict_next_30_days(model_path, scaler_path, data_scaled, time_steps, days, progress_bar=None):
#     model = load_model(model_path)
#     scaler = joblib.load(scaler_path)

#     predictions = []
#     current_data = data_scaled[-time_steps:]  # Use the most recent data to start

#     for _ in range(days):
#         # input_data = current_data.reshape((1, time_steps, data_scaled.shape[1]))
#         input_data = np.reshape(current_data, (1, time_steps, data_scaled.shape[1]))
#         predicted = model.predict(input_data, verbose=0)
#         # predicted = scaler.inverse_transform(predicted_scaled)
#         pred = scaler.inverse_transform(predicted)
#         predictions.append(pred[0])
#         # current_data = np.vstack([current_data[1:], pred])
#         current_data = np.append(current_data[1:], predicted, axis=0)

#         if progress_bar:
#             progress_bar.progress((_ + 1) / days, text="Predicting, please wait...")  # Update progress bar
#     progress_bar.empty()
    
#     return np.array(predictions)

# def predict_next_30_days(model_path, district, scaler_path, data_scaled, time_steps, days, progress_bar=None):
def predict_next_30_days(district, data, time_steps, days, progress_bar=None):
    # model = load_model(model_path)
    # scaler = joblib.load(scaler_path)
    path = snapshot_download("abatejemal/3_Models")
    scaler = joblib.load(f'{path}/weather_models/{district}_scaler.pkl')
    # model_path = snapshot_download("abatejemal/3_Models")
    model = joblib.load(f'{path}/weather_models/{district}_lstm_model.h5')
    # model = load_model(f'{path}/weather_models/{district}_lstm_model.h5')
    data_scaled = scaler.transform(data)
    predictions = []
    current_data = data_scaled[-time_steps:]  # Use the most recent data to start

    for _ in range(days):
        # input_data = current_data.reshape((1, time_steps, data_scaled.shape[1]))
        input_data = np.reshape(current_data, (1, time_steps, data_scaled.shape[1]))
        predicted = model.predict(input_data, verbose=0)
        # predicted = scaler.inverse_transform(predicted_scaled)
        pred = scaler.inverse_transform(predicted)
        predictions.append(pred[0])
        # current_data = np.vstack([current_data[1:], pred])
        current_data = np.append(current_data[1:], predicted, axis=0)

        if progress_bar:
            progress_bar.progress((_ + 1) / days, text="Predicting, please wait...")  # Update progress bar
    progress_bar.empty()
    
    return np.array(predictions)
