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






                        for table_count,table in enumerate(output_tables):

                            for df_iden,df_part in enumerate(split_dataframe(table)):
                                # Initializing variables for adding column and table headers
                                indices = 0
                                add_column = False
                                add_table = False
                                
                                # Checking if dataframe is wide or long
                                is_table_wide = 1 if df_part.shape[1] > 20 else 0
                                
                                # Generating unique table id
                                table_id = table_id_temp + "_" + str(table_count) + "_" + str(df_iden)
                                
                                # Checking if dataframe is not empty
                                if not df_part.empty:
                                    
                                    # Augmenting table data
                                    df_part = table_augmentation(df_part)
                                    
                                    # Removing null columns
                                    nan_value = float("NaN")
                                    df_part.replace("", nan_value, inplace=True)
                                    df_part.dropna(how='all', axis=1, inplace=True)
                                    
                                    # Assigning labels as headers
                                    old_columns = [x.lower() for x in df_part.columns]
                                    new_columns = [fhir_table_mapping.loc[table_name + '_' + x.lower(), "Target_Mapping"] for x in df_part.columns]
                                    
                                    # Generating random numbers to add column and table headers
                                    col_prob1 = [45, 43, 23, 1, 78]
                                    table_prob1 = [12, 43, 64, 76, 22]
                                    col_random_number = np.random.randint(0, 100)
                                    table_random_number = np.random.randint(0, 100)
                                    
                                    # Checking if column header needs to be added
                                    if col_random_number in col_prob1:
                                        add_column = True
                                        
                                    # Checking if table header needs to be added
                                    if table_random_number in table_prob1:
                                        add_table = True
                                    
                                    # Looping through each column of dataframe
                                    for k in range(df_part.shape[1]):
                                        
                                        # Checking if column has any non-NAN values
                                        if len(df_part.iloc[:, k].dropna()) > 0:
                                            
                                            # Creating data list in required format
                                            df1 = df_part.iloc[:, k]
                                            dtype_columns = typesets.CompleteSet().infer_type(df1.dropna())
                                            data_text = str(dtype_columns) + " " + " ".join([str(x) for x in df_part.iloc[:, k].dropna().tolist()])
                                            
                                            # Adding column header if required
                                            if add_column:
                                                data_text = old_columns[k] + " " + data_text
                                                
                                            # Adding table header if required
                                            if add_table:
                                                data_text = table_name + " " + data_text
                                                
                                            # Adding data to data list
                                            if df_part.shape[1] <= 20:
                                                data_list_512.append([table_id, indices, new_columns[k], fhir_coltype_id_dict[new_columns[k].lower()], data_text])
                                            data_list_wide.append([table_id, indices, new_columns[k], fhir_coltype_id_dict[new_columns[k].lower()], data_text])
                                            indices += 1
                                    
                                    # Saving dataframe to CSV
                                    df_part_temp = df_part.copy()
                                    df_part_temp.columns = [x.lower() for x in df_part_temp.columns]
                                    df_part_temp.to_csv(os.path.join(selab_cv_folder_path, table_id + ".csv"), index=False)

# Printing status after completing iteration
iteration += 1
print("Completed iteration number", iteration)

                                    

        
