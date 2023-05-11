
def create_new_cv_data(cv_count):
    validation_list = []
    nsample = 3000
    
    for cv in tqdm.tqdm(cv_count):
        selab_cv_folder_path = os.path.join(selab_saving_folder_path, str(cv))
        os.makedirs(selab_cv_folder_path, exist_ok=True)
        
        print("Create data for the cv:", cv)
        logger.info(f"Create data for the cv: {cv}")
        
        # Create an empty column list
        col_list = []
        
        # Create an empty data list
        data_list_wide = []
        data_list_512 = []
        
        # Run inner loop for 700 times, can be chosen individually
        for iteration in range(1):
            i = 0
            k = 0
            indices = 0
            df_iden = 0
            
            # Logic to do for mimic
            all_files = file_list_generator()
            random.shuffle(all_files)
            
            for file_ind, current_file in enumerate(all_files):
                data_count = []
                
                try:
                    file = current_file[0]
                    table_name = current_file[1]
                    source_name = current_file[2]
                    
                    print(file, table_name, source_name)
                    sample_count = data_count_dict[source_name]
                    nsample = data_count_dict[source_name]
                    
                    data_count.extend([file, table_name, source_name])
                    
                    # loading csv files
                    if file.endswith((".csv", ".parquet")):
                        
                        table_id_temp = table_name + "_cv" + str(cv)
                        
                        logger.info(f"Started reading table: {table_name} from cv: {cv}, having table index: {file_ind} out of {len(all_files)}")
                            
                        elif file.endswith(".parquet"):
                            df = pd.read_parquet(file)
                            df = df.sample(frac=1).reset_index(drop=True)
                            
                            if len(df) > nsample:
                                df = df.sample(nsample, replace=False)
                            else:
                                df = df.sample(nsample, replace=True)
                        
                            print(len(df), "parquet_Data")
                            
                        else:
                            with open(os.path.join(input_data_folder_path, file)) as f:
                                num_rows = sum(1 for row in f)

                            if num_rows >= nsample:
                                random_rows = np.random.choice(np.arange(1, num_rows), size=nsample, replace=False)
                                random_rows[0] = 0
                            else:
                                random_rows = np.random.choice(np.arange(1, num_rows), size=nsample, replace=True)
                                random_rows[0] = 0

                            df = pd.read_csv(file, skiprows=lambda x: x not in random_rows, header=0)
                    
                        df.reset_index(drop=True, inplace=True)
                        print("input_data_row_count", df.shape[0])
                        data_count.append(df.shape[0])
                        print("input_data", df.shape[1])
                        data_count.append(df.shape[1])
                        df = df.loc[:, df.columns.str.lower().isin(fhir_table_mapping[fhir_table_mapping['Table_Name'] == table_name ]["Column_Name"].tolist())]
                        print("mapped_data_count", df.shape[1])
                        data_count.append(df.shape[1])
                        
                        
                        

import os
import glob
import random
import pandas as pd


def file_list_generator(input_files_list, input_data_folder_path, data_read_dict):
    """
    Generates a list of files from given input_files_list and input_data_folder_path.

    Args:
        input_files_list (list): A list of folder names.
        input_data_folder_path (str): The path to the input data folder.
        data_read_dict (dict): A dictionary of folder names and their read status.

    Returns:
        list: A list of files along with their file names, in the following format:
            [[file_path, file_name, folder_name], [file_path, file_name, folder_name], ...]
    """
    all_files = []
    for folder_name in input_files_list:
        if folder_name in data_read_dict.keys():
            if data_read_dict[folder_name] == "file_read":
                # If the folder has been read before
                file_list = glob.glob(os.path.join(input_data_folder_path, folder_name, "*.csv"))
                file_list1 = glob.glob(os.path.join(input_data_folder_path, folder_name, "*.parquet"))
                total_list = file_list + file_list1
                random.shuffle(total_list)

                # Add each file to the all_files list along with its file name and folder name
                for x in total_list:
                    all_files.append([x, x.split("/")[-1][:-4].lower(), folder_name])

            else:
                # If the folder has not been read before
                folder_list = os.listdir(os.path.join(input_data_folder_path, folder_name))
                for i in folder_list:
                    # Look for CSV and Parquet files in the subfolders of the current folder
                    file_paths_csv = glob.glob(os.path.join(input_data_folder_path, folder_name, i, "*.csv"))
                    file_paths_csv1 = glob.glob(os.path.join(input_data_folder_path, folder_name, i, "*", "*.csv"))
                    file_paths_parquet = glob.glob(os.path.join(input_data_folder_path, folder_name, i, "*.parquet"))
                    file_paths_parquet1 = glob.glob(os.path.join(input_data_folder_path, folder_name, i, "*", "*.parquet"))
                    file_paths_parquet_combined = file_paths_parquet + file_paths_parquet1
                    random.shuffle(file_paths_parquet_combined)

                    # Read each Parquet file until one with at least 500 rows is found
                    for ind, f in enumerate(file_paths_parquet_combined):
                        temp = pd.read_parquet(f)
                        if temp.shape[0] >= 500:
                            file_paths_parquet_combined = [file_paths_parquet_combined[ind]]
                            break
                        else:
                            continue

                    # Combine all file paths and choose one at random
                    filename_list = file_paths_csv + file_paths_csv1 + file_paths_parquet_combined
                    if len(filename_list) > 0:
                        random.shuffle(filename_list)
                        all_files.append([filename_list[0], i.lower(), folder_name])

    return all_files

import os
import glob
import random
import pandas as pd

def file_list_generator(input_data_folder_path):
    """
    This function generates a list of files in the specified input folder path and its subdirectories.
    The files are randomly shuffled and stored in a list of lists, with each inner list containing:
    [file path, lowercase filename without extension, parent folder name]

    Args:
    - input_data_folder_path: string, the path to the input data folder

    Returns:
    - all_files: list of lists, where each inner list contains the file path, lowercase filename without extension, 
      and parent folder name
    """

    # Create a list of all the subdirectories and files in the input folder path
    input_files_list = os.listdir(input_data_folder_path)
    all_files = []

    # Loop through each item in the input folder path
    for folder_name in input_files_list:

        # Find all the files with a .csv or .parquet extension in the current subdirectory
        file_list = glob.glob(os.path.join(input_data_folder_path,folder_name,"*.csv"))
        file_list1 = glob.glob(os.path.join(input_data_folder_path,folder_name,"*.parquet"))
        total_list = file_list + file_list1

        # Randomly shuffle the list of files
        random.shuffle(total_list)

        # Append each file path, lowercase filename without extension, and parent folder name to the all_files list
        for x in total_list:
            all_files.append([x,x.split("/")[-1][:-4].lower(),folder_name])

    return all_files
