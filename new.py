import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_new_cv_data(input_data_folder_path, cv_count, nsample=3000):
    """
    Generates cross-validation data by randomly selecting rows from CSV/Parquet files in a directory, stratifying the 
    data, and generating tables based on four strategies.
    
    Args:
    - input_data_folder_path (str): path to directory containing CSV/Parquet files
    - cv_count (int): number of cross-validation folds
    - nsample (int): number of rows to sample from each file
    
    Returns:
    - validation_list (list): a list of lists containing data counts for each fold and table generated
    """
    
    validation_list = []
    
    for cv in tqdm(range(cv_count)):
        
        col_list = []
        data_list_wide = []
        data_list_512 = []
        
        for iteration in range(1):
            i = 0
            k = 0
            indices = 0
            df_iden = 0
            
            # Shuffle files in directory
            all_files = file_list_generator(input_data_folder_path)
            random.shuffle(all_files)
            
            for file_ind, current_file in enumerate(all_files):
                data_count = []

                file_path = current_file[0]
                table_name = current_file[1]
                source_name = current_file[2]
                file_name = current_file[3]
                sample_count = nsample

                data_count.extend([file_path, table_name, source_name])

                if file_path.endswith((".csv", ".parquet")):
                    if file_path.endswith('.csv'):
                        # Read data from file in chunks of 100000 rows
                        chunks = pd.read_csv(file_path, chunksize=100000)
                        dfs = []
                        for chunk in chunks:
                            # Sample nsample rows from each chunk
                            df_sampled = chunk.sample(nsample, replace=True)
                            dfs.append(df_sampled)
                        df = pd.concat(dfs, ignore_index=True)
                        
                    elif file_path.endswith(".parquet"):
                        # Read data from file in chunks of 100000 rows
                        chunks = pd.read_parquet(file_path, chunksize=100000)
                        dfs = []
                        for chunk in chunks:
                            # Sample nsample rows from each chunk
                            if len(chunk) > nsample:
                                df_sampled = chunk.sample(nsample, replace=False)
                            else:
                                df_sampled = chunk.sample(nsample, replace=True)
                            dfs.append(df_sampled)
                        df = pd.concat(dfs, ignore_index=True)
                        
                    else:
                        with open(file_path) as f:
                            num_rows = sum(1 for row in f)

                        # Read data from file in chunks of 100000 rows
                        chunks = pd.read_csv(file_path, chunksize=100000, header=0)
                        dfs = []
                        for i, chunk in enumerate(chunks):
                            if i == 0:
                                # Include the first row of the file in the first chunk
                                chunk = pd.concat([pd.DataFrame([chunk.columns]), chunk], ignore_index=True)
                            if i == len(chunks)-1:
                                # Include the last row of the file in the last chunk
                                chunk = pd.concat([chunk, pd.read_csv(file_path, skiprows=(i+1)*100000, nrows=1, header=None)], ignore_index=True)
                            # Sample nsample rows from each chunk
                            if num_rows >= (i+1)*100000:
                                df_sampled = chunk.sample(nsample, replace=False)
                            else:
                                df_sampled = chunk.sample(nsample, replace=True)
                            dfs.append(df_sampled)
                        df = pd.concat(dfs, ignore_index=True)


                    # Reset dataframe index incase of duplicate rows
                    df.reset_index(drop=True, inplace=True)
                    
                    # Filter columns based on FHIR table mapping
                    df = df.loc[:,df.columns.str.lower().isin(fhir_table_mapping[fhir_table_mapping['table_name'] == file_name ]["column_name"].tolist())]
                    
                    # Stratify data
                    sampler = new_stratified_sampler(sample_count)
                    sampler.fit(df)
                    stratified_df_original = sampler.sample(df)

                    # Get null-removed data
                    null_removed_df = get_null_removed_data(df)

                    # Filter columns in stratified data based on null-removed data
                    stratified_df = stratified_df_original.loc[:,null_removed_df.columns.tolist()]

                    # Generate tables based on four strategies
                    output_tables = table_generation_strategy(stratified_df)
                    
                    # Update data count
                    data_count.append(df.shape[0]) #
