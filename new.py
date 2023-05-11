def create_new_cv_data(input_data_folder_path,cv_count,nsample=3000):


    validation_list = []
    for cv in tqdm.tqdm(cv_count):
        
        # print("Create data for the cv:", cv)
        # logger.info(f"Create data for the cv: {cv}")
        
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
            all_files = file_list_generator(input_data_folder_path)
            random.shuffle(all_files)
            
            for file_ind,current_file in enumerate(all_files):
                data_count = []

                
                file = current_file[0]
                table_name = current_file[1]
                source_name = current_file[2]
                file_name = current_file[3]
                # print(file,table_name,source_name)
                sample_count = nsample
                nsample = nsample
                
                data_count.extend([file, table_name,source_name])
                
                # loading csv files
                if file.endswith((".csv", ".parquet")):
                    
                    #filename = file.split("/")[-1]
                    #table_name = filename[:-4].lower()

                    #if table_name in table_dict.keys():
                
                    table_id_temp = table_name + "_cv" + str(cv) 


                    if file.endswith('.csv'):
                                                # Read the selected rows from the CSV file
                        #print ("Big file processed in if :",file)
                        
                            
                        df_temp = pd.read_csv( file)    
                        df = df_temp.sample(nsample,replace = True)
                        
                        
                    elif file.endswith(".parquet"):
                        df = pd.read_parquet(file)
                        df = df.sample(frac=1).reset_index(drop=True)
                        if len(df) > nsample:
                            
                            df = df.sample(nsample, replace=False)
                        else:
                            df = df.sample(nsample, replace=True)
                            
                        
                        print(len(df),"parquet_Data")
                        
                    
                    
                    else:

                        # Calculate the number of rows in the CSV file
                        with open(os.path.join(input_data_folder_path, file)) as f:
                            num_rows = sum(1 for row in f)

                        # Choose a random set of row indices
                        if num_rows >= nsample:
                            random_rows = np.random.choice(np.arange(1, num_rows), size=nsample, replace=False)
                            random_rows[0]=0
                        else:
                            random_rows = np.random.choice(np.arange(1, num_rows), size=nsample, replace=True)
                            random_rows[0]=0

                        #skiprows=lambda x: x not in random_rows

                        # Read the selected rows from the CSV file
                        df = pd.read_csv(file, skiprows=lambda x: x not in random_rows, header=0)

                
                        
            

                    # reset dataframe index incase of duplicate roes
                
                    df.reset_index(drop=True, inplace=True)
                    #df = df.loc[:, df.columns.str.lower().isin(table_dict[table_name.lower()].keys())]  
                    print("input_data_row_count", df.shape[0])
                    data_count.append(df.shape[0])
                   
                    print("input_data", df.shape[1])
                    data_count.append(df.shape[1])
                    print (df.columns)
                    
                    df = df.loc[:,df.columns.str.lower().isin(fhir_table_mapping[fhir_table_mapping['table_name'] == file_name ]["column_name"].tolist())]
                    
                    # print("mapped_data_count", df.shape[1])
                    data_count.append(df.shape[1])
                    #print ("After",df.shape)  
                    
                    print ("Shape of df is :",df.shape)
                    
                    # create stratified data
                    sampler = new_stratified_sampler(sample_count)
                    sampler.fit(df)
                    # Generate a stratified random sample of the input dataframe
                    stratified_df_original = sampler.sample(df)
                    print(stratified_df_original.shape[1], df.shape[1], "original stratified df")

                    
                    
                    # null removed df
                    null_removed_df = get_null_removed_data(df)
                    print("mull removed data", df.shape[1])
                    data_count.append(df.shape[1])
                    '''
                    df_original_col_list = [x.lower() for x in df.columns]
                    tabert_df["doduo_included"] = tabert_df["original_table_name"].apply(lambda x : 1 if x in df_original_col_list  else 0)
                    data_count.append(tabert_df["doduo_included"].sum())
                    tabert_df_list.append(tabert_df)
                    '''
                    
                    
                    # remove null columns from stratified df for doduo
                    stratified_df = stratified_df_original.loc[:,null_removed_df.columns.tolist()]
                    
                    

                    
                    data_count.append(stratified_df.shape[0])
                    
                    data_count.append(stratified_df.shape[1])
                    


                    # generate strategic tables based on four strategies :  output 40 tables

                    output_tables = table_generation_strategy(stratified_df)
                    print("stratgeized_table_count", len(output_tables))
                    data_count.append(len(output_tables))
                    validation_list.append(data_count)
