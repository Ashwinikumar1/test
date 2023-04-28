

#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

import sys
import os
import shutil
import pandas as pd
sys.path.append('../')  # add parent directory to the module search path
from fastapi import FastAPI
from utils.utils import *  # import utility functions
from data_loader.data_preprocessing import *  # import data preprocessing functions
from configs.config import config  # import configuration settings

# load the configuration
config = config('/mnt/batch/tasks/shared/LS_root/mounts/clusters/optimus-aml-gpu/code/Users/ashwini_kumar/Optimus_Data_Transformer_pipeline/configs/config.yaml')

app = FastAPI()  # create a FastAPI instance

@app.post("/transform_data")  # HTTP POST method decorator for the transform_data() function
async def transform_data(raw_data_directory: str):
    """
    Applies data transformation to the raw data and generates canonical input data.

    Args:
    raw_data_directory (str): path to the directory containing raw data files.

    Returns:
    dict: A dictionary containing the status and message of the transformation process.
    """
    global doduo_embeddings_list 
    doduo_embeddings_list = list()  # create a list to store doduo embeddings for all input files

    # Define Output Location
    intermediate_data_location = config.OUTPUT_DIR  # get the output directory path from configuration settings

    # Create directory if it does not exist
    os.makedirs(intermediate_data_location, exist_ok=True)

    # Create a safe location for canonical input data
    canonical_save_location = os.path.join(intermediate_data_location, "canonical_input_data")

    # Empty the save location if it already exists
    if os.path.exists(canonical_save_location):
        shutil.rmtree(canonical_save_location)

    # Create a new directory for canonical input data
    os.makedirs(canonical_save_location, exist_ok=True)

    print("Canonical Save Location :", canonical_save_location)

    # Get a list of all CSV files in the specified directory
    raw_data_file_list = [x for x in os.listdir(raw_data_directory) if x.endswith(".csv")]

    for file_name in raw_data_file_list:
        table_name = file_name.split(".csv")[0]  # extract table name from file name
        input_df = pd.read_csv(os.path.join(raw_data_directory, file_name))  # read input CSV file
        canonical_data_location = os.path.join(canonical_save_location, table_name)  # create directory for canonical input data

        # Create directory if it does not exist
        os.makedirs(canonical_data_location, exist_ok=True)

        # Apply data transformer to the input data and get the output
        output = input_table_annotations(input_df, table_name)

        # Store mapping of raw data columns to canonical data columns in a CSV file
        output[0].to_csv(os.path.join(canonical_data_location, "data_mapping.csv"), index=False)

        # Store stratified data in a CSV file
        output[1].to_csv(os.path.join(canonical_data_location, "stratified_data.csv"), index=False)

        # Add the doduo embeddings for the current input file to the list
        doduo_embeddings_list.append(output[2])

    # Combine the doduo embeddings for all input files into a single DataFrame
    doduo_embeddings_df = pd.concat(doduo_embeddings_list, axis=0)
    doduo_embeddings_df.reset_index(drop=True, inplace=True)

    # Save the doduo embeddings to a pickle file
    doduo_embeddings_df.to_pickle(os.path.join(canonical_save_location, "data_transformer_embeddings.pkl"))

    # Delete to clear the memory
    del doduo_embeddings_list 
    # Return the status
    return {"status": "success", "message": f"Data transformation completed successfully. Output file path: {canonical_save_location}"}


curl -X POST -H "Content-Type: application/json" -d '{"raw_data_directory": "/path/to/raw/data"}' http://localhost:8000/transform_data


Install FastAPI: pip install fastapi

Install uvicorn server: pip install uvicorn