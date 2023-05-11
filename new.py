

import os
import pandas as pd


def get_table_columns(folder_path: str) -> pd.DataFrame:
    """
    Extracts the column names from CSV and Parquet files in a folder and its subfolders.

    Args:
        folder_path (str): Path to the folder to search for files.

    Returns:
        A Pandas DataFrame with two columns: "table_name" and "column_name". Each row
        represents a column in a file in the specified folder and its subfolders.

    """
    table_columns = []
    # Walk through all subfolders in the specified directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                # If it's a CSV file, read the file using Pandas and extract the column names
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, nrows=100)  # Read only the first 100 rows
                table_name = file[:-4]
                columns = [col.lower() for col in df.columns]
                # Add the table name and column name to the list
                for col in columns:
                    table_columns.append((table_name, col))
            elif file.endswith('.parquet'):
                # If it's a Parquet file, read the file using Pandas and extract the column names
                file_path = os.path.join(root, file)
                df = pd.read_parquet(file_path, nrows=100)  # Read only the first 100 rows
                table_name = file[:-8]
                columns = [col.lower() for col in df.columns]
                # Add the table name and column name to the list
                for col in columns:
                    table_columns.append((table_name, col))
    # Create a Pandas DataFrame from the list of table names and column names
    df_table_columns = pd.DataFrame(table_columns, columns=['table_name', 'column_name'])
    return df_table_columns


import pandas as pd
from typing import Dict
import os


def map_target_representation(df_table_columns: pd.DataFrame, mapping_file_csv: str) -> Dict:
    """
    Inner joins a DataFrame of table and column names with a mapping file on 'table_name'
    and 'column_name', and adds a 'target_mapping' column from the mapping file. Then, it
    creates a dictionary of mapped target representations where the key is the target mapping
    and the value is a sequence from 0 and so on.

    Args:
        df_table_columns (pd.DataFrame): A Pandas DataFrame with two columns: "table_name"
            and "column_name". Each row represents a column in a file in a specified folder
            and its subfolders.
        mapping_file_csv (str): Path to the CSV file containing the mapping information. The
            file must contain 'table_name', 'column_name', and 'target_mapping' columns.

    Returns:
        A dictionary where the keys are the unique target mappings from the mapping file and
        the values are a sequence from 0 and so on.

    Raises:
        ValueError: If fewer than 95% of the columns in the DataFrame can be mapped to target
            representations.

    """
    # Check if the mapping file has been modified since the last time the function was called
    mapping_mod_time = os.path.getmtime(mapping_file_csv)
    if hasattr(map_target_representation, 'last_mod_time') and map_target_representation.last_mod_time == mapping_mod_time:
        return map_target_representation.target_dict
    
    # Load the mapping file into a DataFrame
    mapping_df = pd.read_csv(mapping_file_csv)
    # Inner join the two DataFrames on 'table_name' and 'column_name'
    merged_df = pd.merge(df_table_columns, mapping_df, on=['table_name', 'column_name'], how='inner')
    # Create a dictionary of mapped target representations
    target_dict = {target: i for i, target in enumerate(merged_df['target_mapping'].unique())}
    # Add a 'target_index' column to the merged DataFrame
    merged_df['target_index'] = merged_df['target_mapping'].map(target_dict)
    # Calculate the percentage of columns that were successfully mapped
    mapped_pct = len(merged_df) / len(df_table_columns)
    # Raise an error if fewer than 95% of columns were mapped
    if mapped_pct < 0.95:
        raise ValueError("Very Few Matches Found")
    
    # Save the dictionary and last modified time as attributes of the function
    map_target_representation.target_dict = target_dict
    map_target_representation.last_mod_time = mapping_mod_time
    
    return target_dict



import numpy as np
import pandas as pd

class new_stratified_sampler:
    """
    A class for sampling data from a dataframe in a stratified way.
    
    Attributes:
    target_count (int): The desired number of samples to generate.
    values_freq (dict): A dictionary that stores the frequency of values for each numeric column in the dataframe.
    """
    
    def __init__(self, target_count):
        """
        Initializes a new instance of the new_stratified_sampler class.
        
        Args:
        target_count (int): The desired number of samples to generate.
        """
        self.target_count = target_count
        self.values_freq = {}
        
    def fit(self, df):
        """
        Fits the sampler to a given dataframe by computing the frequency of values for each numeric column.
        
        Args:
        df (pandas.DataFrame): The dataframe to fit the sampler to.
        """
        for column in df.select_dtypes(include=[np.number]):
            values_freq = df[column].value_counts()
            self.values_freq[column] = values_freq
            
    def sample(self, df):
        """
        Samples the given dataframe in a stratified way according to the fitted values_freq dictionary.
        
        Args:
        df (pandas.DataFrame): The dataframe to sample.
        
        Returns:
        pandas.DataFrame: A dataframe containing the sampled data.
        """
        
        # Initialize an empty dataframe for the sampled data
        samples_df = pd.DataFrame()
        
        # Sample from each column in the dataframe
        for column in df.columns:
            
            # Replace empty string values with NaN
            nan_value = float("NaN")
            df[column].replace("", nan_value, inplace=True)
            
            if df[column].isnull().all():
                # If the column is empty, pass directly
                samples_df[column] = df[column]
                
            elif pd.api.types.is_bool_dtype(df[column].dtype):
                # If the column is boolean, sample randomly
                sampled_values = np.random.choice(df[column].dropna(), size=self.target_count, replace=True)
                samples_df[column] = pd.Series(sampled_values)
                
            elif pd.api.types.is_numeric_dtype(df[column].dtype):
                # If the column is numeric, sample in a stratified way
                values_freq = self.values_freq[column]
                values_prob = values_freq / values_freq.sum()
                sampled_values = np.random.choice(values_freq.index, size=self.target_count, p=values_prob)
                samples_df[column] = pd.Series(sampled_values)
                
            else:
                # If the column is categorical, sample randomly
                sampled_values = np.random.choice(df[column].dropna(), size=self.target_count, replace=True)
                samples_df[column] = pd.Series(sampled_values)
        
        return samples_df
    
    def generate_samples(self, df):
        """
        Generates samples iteratively for each 100000 rows of the given dataframe and combines them into a single output.
        
        Args:
        df (pandas.DataFrame): The dataframe to sample.
        
        Returns:
        pandas.DataFrame: A dataframe containing the sampled data.
        """
        samples_list = []
        for i in range(0, len(df), 100000):
            subset_df = df.iloc[i:i+100000]
            samples_df = self.sample(subset_df)
            samples_list.append(samples_df)
        combined_samples_df = pd.concat(samples_list)
        return combined_samples_df
