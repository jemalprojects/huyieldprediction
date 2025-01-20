import os
import sys
import time
import json
import urllib3
import requests
import multiprocessing
import pandas as pd
from datetime import datetime
import streamlit as st
from huggingface_hub import HfApi, upload_file, login
import io
urllib3.disable_warnings()

# Destination folder to save the data
dest_dir = 'WeatherData/'
# File path containing location data of districts with their latitudes and longitudes
locations_file_path = './districtlevel_locations.csv'  # Modify this CSV file to have district locations
# File path for the parameter to download from NASA website, add or remove parameters from this file 
parameters_file_path = './params.csv'
# start_date = '20040101'  # Format YYYYMMDD
# start_date=datetime.datetime.strptime(start_date, "%m/%d/%Y").strftime("%Y%m%d")
# end_date = '20241220'    # Format YYYYMMDD
# last_date=pd.read_csv(dest_dir+'Anchar.csv')
# most_recent_date = last_date['date'].max()
# start_date=datetime.strptime(most_recent_date, "%Y-%m-%d").strftime("%Y%m%d")
start_date = '20040101'  # Format YYYYMMDD
end_date = datetime.today().strftime('%Y%m%d')

import pandas as pd
import io
from huggingface_hub import login, upload_file

def upload_dataframe_to_huggingface(dataframe, data_filename):
    # Hugging Face token
    token = st.secrets["huggingface_TOKEN"]
    
    # Log in to Hugging Face
    login(token=token)
    
    # Repository details
    repo_name = "2_Data"
    username = "abatejemal"
    repo_id = f"{username}/{repo_name}"  # Full repository identifier
    
    # Convert the DataFrame to a binary CSV file in-memory
    csv_buffer = io.BytesIO()
    dataframe.to_csv(csv_buffer, index=False, encoding='utf-8')  # Write CSV as bytes
    csv_buffer.seek(0)  # Reset buffer pointer to the start
    
    # Upload the in-memory binary CSV file to the Hugging Face repository
    upload_file(
        path_or_fileobj=csv_buffer,   # In-memory binary CSV file
        path_in_repo=data_filename,  # Name of the file in the repository
        repo_id=repo_id,             # Repository ID
        repo_type="dataset",         # Specify repository type as dataset
    )



def download_function(collection):
    request, filepath, row = collection
    response = requests.get(url=request, verify=False, timeout=30.00).json()

    if 'properties' in response and 'parameter' in response['properties']:
        data = response['properties']['parameter']
        df = pd.DataFrame(data)
        # Generate the date range from '2004-01-01' to '2017-12-30'
        end_date1=datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
        date_range = pd.date_range(start=start_date, end=end_date1)
        df['date'] = date_range[:len(df)]  # Ensure the date range matches the DataFrame length
        df['region'] = row['Region']
        df['zone'] = row['Zone']
        df['district'] = row['district']
        df['weredacode'] = row['WeredaCode']
        df.to_csv("2_Data/"+filepath, index=False)
        district=df['district']
        # data_filename=f"WeatherData/{district}.csv"
        upload_dataframe_to_huggingface(df, filepath)
    else:
        print(f"Unexpected response format for request: {request}")

class Process:
    def __init__(self):
        self.processes = 5  # Please do not go more than five concurrent requests.
        self.request_template = (
            r"https://power.larc.nasa.gov/api/temporal/daily/point?"
            "parameters={parameters}&community=AG&longitude={longitude}&latitude={latitude}"
            "&start={start_date}&end={end_date}&format=JSON&time-standard=UTC"
        )
        # self.filename_template = "{dir}{loc_name}_Lat_{latitude}_Lon_{longitude}.csv"
        self.filename_template = "{dir}{loc_name}.csv"
        # self.filename_template = "{dir}{loc_name}.csv"
        self.times = {}

    def execute(self, dir, locations_file_path, parameters_file_path, start_date, end_date):
        locations = pd.read_csv(locations_file_path)
        parameters = ','.join(pd.read_csv(parameters_file_path)['Param_Code'].tolist())
        dir = dest_dir
        start_time = time.time()

        requests_list = []
        address=[]
        for _, row in locations.iterrows():
            loc_name = row['district']
            WeredaCode = row['WeredaCode']
            wereda = row['district']
            Zone = row['Zone']
            Region = row['Region']
            latitude, longitude = row['lat'], row['lng']
            
            request = self.request_template.format(
                parameters=parameters, latitude=latitude, longitude=longitude, start_date=start_date, end_date=end_date
            )
            # filename = self.filename_template.format(dir=dir, loc_name=loc_name, latitude=latitude, longitude=longitude)
            filename = self.filename_template.format(dir=dir, loc_name=loc_name, WeredaCode=WeredaCode, Zone=Zone, Region=Region)
            requests_list.append((request, filename, row))

        requests_total = len(requests_list)

        pool = multiprocessing.Pool(self.processes)
        x = pool.imap_unordered(download_function, requests_list)
        progress_bar = st.progress(0, text="0%")
        for i, _ in enumerate(x, 1):
            progress_bar.progress(i/requests_total, text=f'\rExporting {i/requests_total:.2%}')
            sys.stderr.write(f'\rExporting {i/requests_total:.2%}')

        self.times["Total Script"] = round((time.time() - start_time), 2)

        print("\n")
        print("Total Script Time:", self.times["Total Script"])

def main():
    os.makedirs(dest_dir, exist_ok=True)

    # District data with lat and long
    Process().execute(
        dir=dest_dir,
        locations_file_path=locations_file_path,
        parameters_file_path=parameters_file_path,
        start_date=start_date,
        end_date=end_date
    )

if __name__ == "__main__":
    main()
