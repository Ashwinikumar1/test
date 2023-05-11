
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