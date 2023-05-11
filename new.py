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
                                indices = 0
                                add_column = False
                                add_table = False

                                is_table_wide = 1 if df_part.shape[1] > 20 else 0

                                table_id = table_id_temp + "_" + str(table_count) + "_" + str(df_iden)

                                
                                if not df_part.empty:

                                    

                                    df_part = table_augmentation(df_part)
                                    
                                    
                                    # remove null columns
                                    
                                    nan_value = float("NaN")
                                    df_part.replace("", nan_value, inplace=True)

                                    df_part.dropna(how='all', axis=1, inplace=True)
                                    

                                    # add label as header to df
                                    
                                    
                                    # assining labels as headers
                                    
                                    old_columns = [x.lower() for x in df_part.columns]
                                   
                                    new_columns = [fhir_table_mapping.loc[ table_name + '_' + x.lower(),"Target_Mapping"] for x in df_part.columns]
                                    #df_part.columns = new_columns
                                    #df_part["flag_table_wide"] = is_table_wide
                                    #print(df_part.columns)
                                    
                                    # add column name and table name with probablity
                                    
                                    col_prob1 = [45,43,23,1,78]
                                    table_prob1 = [12,43,64,76,22]
                                    col_random_number = np.random.randint(0,100)
                                    table_random_number = np.random.randint(0,100)
                                    
                                    if col_random_number in col_prob1:
                                        add_column = True
                                    
                                    if table_random_number in table_prob1:
                                        add_table = True
                                        
                                    
                                    
                                    
                                    
                                    #print(len(df_part.columns) , "87997")
                                    #print(df_part.shape[1], "0808")
                                    #df_part.to_csv("testting.csv", index=False)
                                    for k in range(df_part.shape[1]):
                                        
                                        # Replace all blanks with NaN
                                        #print(df_part.iloc[:, k])
                                        
                                        # need to check
                                    
                                        #df_part.iloc[:, k].replace('', np.nan, inplace=True)
                                        
                                        
                                        
                                            
                                        # If at least one value in column is NOT NAN
                                        if len(df_part.iloc[:, k].dropna()) > 0:
                                            # Create a data list in the required format
                                            df1 = df_part.iloc[:, k]
                                            dtype_columns = typesets.CompleteSet().infer_type(df1.dropna())
                                            #print(fhir_coltype_id_dict[df_part.columns[k].lower()])

                                            #data_list.append([table_id,
                                            #                indices,
                                            #                table_dict[table_name.lower()][df_part.columns[k].lower()].lower(),
                                            #                fhir_coltype_id_dict[table_dict[table_name.lower()][df_part.columns[k].lower()].lower()],
                                            #                str(dtype_columns) + " " + " ".join([str(x) for x in df_part.iloc[:, k].dropna().tolist()])])
                                            
                                            
                                            data_text = str(dtype_columns) + " " + " ".join([str(x) for x in df_part.iloc[:, k].dropna().tolist()])
                                            #print(data_text)
                                            if add_column :
                                                            data_text = old_columns[k] + " " + data_text 
                                                                                            
                                            if add_table:
                                                            data_text = table_name + " " + data_text
                                                                                            
                                                                                            
                                                
                                            if df_part.shape[1] <= 20:
                                                                    data_list_512.append([table_id, indices, new_columns[k],fhir_coltype_id_dict[new_columns[k].lower()] ,data_text])
                                                                                            
                                    
                                                            
                                            data_list_wide.append([table_id, indices, new_columns[k],fhir_coltype_id_dict[new_columns[k].lower()] ,data_text])
                                            
                                            
                                            
                                            indices = indices + 1
                                        

                                
                                    df_part_temp = df_part.copy()
                                    df_part_temp.columns  = [x.lower() for x in df_part_temp.columns]
                                    '''
                                    if not df_part_temp.empty:
                                        
                                    
                                        if add_column:


                                                    new_df = pd.DataFrame(old_columns).T
                                                    new_df.columns = df_part_temp.columns
                                                    df_part_temp = pd.concat([new_df, df_part_temp], axis= 0)

                                        if add_table:
                                                    print("inside table append", table_name)
                                                    new_df = pd.DataFrame([table_name] * len(df_part_temp.columns)).T
                                                    new_df.columns = df_part_temp.columns                                             
                                                    df_part_temp = pd.concat([new_df, df_part_temp], axis= 0)

                                    
                                                                                        
                                    '''                                                   
                                    #print(df_part_temp.isna().sum(), "final_table")
                                    
                                    df_part_temp.to_csv(os.path.join(selab_cv_folder_path ,table_id + ".csv"), index=False)
                                    
                except Exception as e :
                    print(current_file[0], "error")
                    print(e)
                    
                    
                    
                    logger.info("error")
                    logger.info(current_file[0])
                    
                    
                    continue                
                                
                                
            iteration = iteration + 1
                # Print iterations just to see progress        
               
            print ("Completed iteration number",iteration)
        
        # tabert mapping df
        #print("tabert_list_length", len(tabert_df_list))
        #tabert_df_list = pd.concat(tabert_df_list, axis=0)
        #tabert_df_list.to_csv(os.path.join(tabert_saving_folder_path, "cv_" + str(cv) + ".csv" ), index=False)
            
        # Convert the list to dataframe with all 5 required columns
        df_wide = pd.DataFrame(data_list_wide, columns=["table_id", "col_idx", "class", "class_id", "data"])
        df_512  = pd.DataFrame(data_list_512, columns=["table_id", "col_idx", "class", "class_id", "data"])                                                                        
        print ("Dataframe Created")


        # # Shuffle the table_id column
        # df['table_id'] = np.random.permutation(df['table_id'])
        # df = df.sort_values(by='table_id')

        # # Sort the col_idx column in ascending order for each table_id value
        # df = df.sort_values(by=['table_id', 'col_idx'])
        # df = df.reset_index(drop=True)
        

        #data_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/aswinivm2gpu16/code/Users/ashwini_kumar/Optimus_Training_v2/lm-annotation-model/data/"
        # Store the data as csv in the required location
        df_wide.to_csv(os.path.join(doduo_wide_saving_folder_path , "msato_cv_{}.csv".format(cv)), index=False)
        df_512.to_csv(os.path.join( doduo_small_saving_folder_path, "msato_cv_{}.csv".format(cv)), index=False)
        # pd.read_csv(data_path + "msato_cv_{}.csv".format(cv))
        # print message CV is completed
        #validation_df = pd.DataFrame(validation_list)
        #validation_df.columns  = ["file_path", "table_name", "source", "input_file_row_count", "input_file_col_count","mapped_data_col_count","tabert_mapped_column_count","null_removed_col_count","doduo_included_count","sample_row_count", "sample_column_count", "strategized_table_count"]
        #validation_df.to_csv(os.path.join(saving_folder_path , "validation_cv_{}.csv".format(cv)), index=False)
        logger.info(f"This cv is complete : {cv}")    
        
