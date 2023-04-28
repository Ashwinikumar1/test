
# Date : 28th April 2023
# Objective : Define Utility Function which are use for Data Preprocessing

'''
Function Defined:
    majority_voting_with_probs : Majority Voting class function takes multiple lables and probs as Input
                                 It ensures that we get one prediction for each columns no matter how many iteration we run

    columns_annotations : Column Annotations function ensures that we take 16 random rows each time and run Doduo annotate on each sample.

'''

import visions as v
#print(v.__version__)
from visions import typesets
# Load the required packages and actual inference Code
import argparse
import visions
import pandas as pd
import numpy as np
import os
import logging
import json
import numpy
import joblib
import time
from flask import Flask, make_response, request
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle
import random
from collections import Counter
from itertools import islice, cycle
import warnings
from itertools import cycle, takewhile, dropwhile
import itertools
warnings.filterwarnings("ignore")
import logging
import sys
from statistics import mean, mode
from pandas.util.testing import assert_frame_equal
import shutil

#  stratified sample class
class StratifiedSampler:
    def __init__(self, target_count):
        self.target_count = target_count
        self.values_freq = {}
    def fit(self, df):
        for column in df.select_dtypes(include=[np.number]):
            values_freq = df[column].value_counts()
            self.values_freq[column] = values_freq
            
    def sample(self, df):
        # Initialize an empty dataframe for the sampled data
        samples_df = pd.DataFrame()     
        # Sample from each column in the dataframe
        for column in df.columns:
            # Skip empty columns
            
            if df[column].isnull().all() or not pd.api.types.is_numeric_dtype(df[column].dtype) or pd.api.types.is_bool_dtype(df[column].dtype):
                continue
            # If the column is numeric, sample in a stratified way
            values_freq = self.values_freq[column]
            values_prob = values_freq / values_freq.sum()
            sampled_values = np.random.choice(values_freq.index, size=self.target_count, p=values_prob)
            samples_df[column] = pd.Series(sampled_values)
        
        # Sample non-numeric columns randomly with replacement
        for column in df.columns:
            if df[column].isnull().all() or pd.api.types.is_numeric_dtype(df[column].dtype):
                
                continue
            
            sampled_values = np.random.choice(df[column].dropna(), size=self.target_count, replace=True)
            samples_df[column] = pd.Series(sampled_values)
            
        
         
        # Sample boolean columns  randomly with replacement
        
        for column in df.columns:
            if pd.api.types.is_bool_dtype(df[column].dtype):
    
                sampled_values = np.random.choice(df[column].dropna(), size=self.target_count, replace=True)
                samples_df[column] = pd.Series(sampled_values)

        
        return samples_df



class new_stratified_sampler:
    def __init__(self, target_count):
        self.target_count = target_count
        self.values_freq = {}
    def fit(self, df):
        for column in df.select_dtypes(include=[np.number]):
            values_freq = df[column].value_counts()
            self.values_freq[column] = values_freq
            
    def sample(self, df):
        # Initialize an empty dataframe for the sampled data
        samples_df = pd.DataFrame()
        
        
        
        
        # Sample from each column in the dataframe
        for column in df.columns:
            nan_value = float("NaN")
            df[column].replace("", nan_value, inplace=True)
            
            if df[column].isnull().all():
                # empty column pass directly
                
                samples_df[column] = df[column]
                
            elif pd.api.types.is_bool_dtype(df[column].dtype):
                # boolean type
                sampled_values = np.random.choice(df[column].dropna(), size=self.target_count, replace=True)
                samples_df[column] = pd.Series(sampled_values)
                
            elif pd.api.types.is_numeric_dtype(df[column].dtype):
                
                # If the column is numeric, sample in a stratified way
                values_freq = self.values_freq[column]
                values_prob = values_freq / values_freq.sum()
                sampled_values = np.random.choice(values_freq.index, size=self.target_count, p=values_prob)
                samples_df[column] = pd.Series(sampled_values)
            
            
                
            else:
                # categorical data
                sampled_values = np.random.choice(df[column].dropna(), size=self.target_count, replace=True)
                samples_df[column] = pd.Series(sampled_values)
    

        
        return samples_df




def get_null_removed_data(temp_df):
    
    '''
    Function to identify null columns in a table and remove based on the emptiness threshold calculated based on the number of rows
    ''' 

    # null removal table df based on number of rows, threshold varies
    null_removal_table_df = pd.DataFrame([[0,1000,0.9],
        [1001,10000,0.95],
        [10001,100000,0.99],
        [100001,100000000,1,]])
    null_removal_table_df.columns = ["min_table_length", "max_table_length","remove_threshold"]  
    count = 0
    row_count = len(temp_df)
    null_removal_table_df = null_removal_table_df.copy()
    # identify threshold based on row count
    for a,b in zip(null_removal_table_df["min_table_length"], null_removal_table_df["max_table_length"]):
        
        if row_count >= a and row_count<=b:
            break
        else:
            count = count+1
    
    remove_threshold = null_removal_table_df.loc[count,"remove_threshold"]
    
    df = temp_df.copy()
    df = df.replace([' ','','NULL'],np.nan)
    result = df.isna().mean()
    # select data which is less than below less than input mising percentage
    df = df.loc[:,result < remove_threshold]
    return df

def sliding_window(self,iterable,size, overlap= 0):
    '''
    This is a utility function which return tuple of list based on overlapping windows
    '''
    start = 0
    end = size
    step = size - overlap
    length = len(iterable)
    while end < length:
        yield tuple(iterable[start:end])
        start += step
        end += step
    yield tuple(iterable[start:])


def table_generation_strategy(df):
    '''
    This function create list of tables based on permutation stratigies
    input to table : original dataframe
    output : list of generated tables
    '''

    # strategies table based on number of columns for repeating specific stratigies
    table_df = pd.DataFrame([[0,49,5,5],
        [50,99,8,8],
        [100,999999999999,20,10]])
    table_df.columns = ["min_col_count","max_col_count","strategy3_iteration_count","strategy4_iteration_count"]

    
    output_df = []
    table_df = table_df.copy()

    # check if no of columns less than 20, then input table only returned without any additional stratigies
    if df.shape[1] <=20:
        output_df.append(df)
        return output_df

    # identify repetation strategy count based on no of columns
    count = 0
    column_count = len(df.columns)
    for a,b in zip(table_df["min_col_count"], table_df["max_col_count"]):

        if column_count >= a and column_count<=b:
            break
        else:
            count = count+1


    strategy3_iteration_count = table_df.loc[count,"strategy3_iteration_count"]
    strategy4_iteration_count = table_df.loc[count,"strategy4_iteration_count"]






    # strategy1 directly pass large size table
    output_df.append(df)

    # additional strategies applicable for only large tables, 

    if df.shape[1] > 20:

        # strategy2 :  sliding window of 20 with overlap of 10

        sliding_window_list = list(self.sliding_window(range(df.shape[1]), 20,10))
        for i in sliding_window_list:
            col_indice = [x for x in list(i) if x < df.shape[1]]
            #print(col_indice, "****")
            output_df.append(df.iloc[:, col_indice])




        # strategy3 : select 20 pairs of (10,10) consecutive columns
        cylical_list = CyclicalList(range(df.shape[1]))

        for i in range(strategy3_iteration_count):

            random_indics = [ 9*i + x for i, x in enumerate(sorted(random.sample(range(df.shape[1]), 2)))]

            indice_1 = cylical_list[random_indics[0]:random_indics[0]+10 ]
            indice_2 = cylical_list[random_indics[1]:random_indics[1]+10 ]
            indice_1 = list(set(indice_1))
            indice_2 = list(set(indice_2))
            indice_1 = list(set(indice_1).difference(set(indice_2)))

            # remove in case any overlapping columns


            df_part_a =   df.iloc[:,indice_1 ]
            df_part_b =   df.iloc[:,indice_2 ]

            merged = pd.concat([df_part_a,df_part_b], axis=1)
            output_df.append(merged)


    return output_df
