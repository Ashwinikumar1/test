#!/usr/bin/env python
# coding: utf-8

import sys
import os
import shutil
import pandas as pd
sys.path.append('../')
from utils.utils import *
from data_loader.data_preprocessing import *
from configs.config import config
import argparse

# load the configuration
config = config('/mnt/batch/tasks/shared/LS_root/mounts/clusters/optimus-aml-gpu/code/Users/ashwini_kumar/Optimus_Data_Transformer_pipeline/configs/config.yaml')

def data_transformer_list_parser(raw_data_directory):
    """
    Applies data transformation to the raw data and generates canonical input data.

    Args:
    raw_data_directory (str): path to the directory containing raw data files.

    Returns:
    None
    """
    global doduo_embeddings_list 
    doduo_embeddings_list = list()

    # Define Output Location
    intermediate_data_location = config.OUTPUT_DIR
    # Create directory if it does not exist
    os.makedirs(intermediate_data_location, exist_ok=True)
    # Create a safe location
    canonical_save_location  = os.path.join(intermediate_data_location,"canonical_input_data")

    # Empty the save location
    if os.path.exists(canonical_save_location):
        shutil.rmtree(canonical_save_location)

    os.makedirs(canonical_save_location,exist_ok=True)

    print ("Canonical Save Location :",canonical_save_location)
    # raw_data_file_list = [x for x in os.listdir(config.INPUT_DIR) if x.endswith(".csv")]
    raw_data_file_list = [x for x in os.listdir(raw_data_directory) if x.endswith(".csv")]

    for file_name in raw_data_file_list:
        table_name = file_name.split(".csv")[0]
        input_df = pd.read_csv(os.path.join(raw_data_directory, file_name))
        canonical_data_location = os.path.join(canonical_save_location, file_name.split(".csv")[0])
        os.makedirs(canonical_data_location, exist_ok=True)

        # data transformer output
        output = input_table_annotations(input_df, table_name)

        # store mapping 
        output[0].to_csv(os.path.join(canonical_data_location, "data_mapping.csv"), index=False)

        # store stratified data
        output[1].to_csv(os.path.join(canonical_data_location, "stratified_data.csv"), index=False)

        # store pickle file for doduo embeddings on the data
        doduo_embeddings_list.append(output[2])

    doduo_embeddings_df = pd.concat(doduo_embeddings_list, axis=0)
    doduo_embeddings_df.reset_index(drop=True, inplace=True)
    doduo_embeddings_df.to_pickle(os.path.join(canonical_save_location, "data_transformer_embeddings.pkl"))
    del doduo_embeddings_list 

# Add the main function
def main():
    """
    Parses command-line arguments and calls the data_transformer_list_parser function.

    Args:
    None

    Returns:
    None
    """
    parser = argparse.ArgumentParser(description='Data Transformer')
    parser.add_argument('--raw_data_directory', type=str, help='Path to raw data directory', required=True)
    args = parser.parse_args()
    raw_data_directory = args.raw_data_directory
    data_transformer_list_parser(raw_data_directory)


if __name__ == "__main__":
    main()
