#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8
import subprocess
import sys
import os
import shutil
import pandas as pd
import sys
import glob
import tqdm
import yaml
import argparse
import torch
import IProgress
import ipywidgets
sys.path.append('../')

from data_loader.data_preprocessing import *
from configs.config import config
# from models.doduo.train_multi import *
# from models.doduo.predict_multi import *

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Optimus Training Pipeline')
parser.add_argument('--input_dir', type=str, help='Path to the input data directory')
parser.add_argument('--mapping_file', type=str, help='Path to the mapping file')
args = parser.parse_args()

# Load the configuration
config_path = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/optimus-aml-gpu/code/Users/ashwini_kumar/optimus_training_pipeline/data_transformer_training_pipeline/configs/config.yaml'
with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)

# Update the input_dir and mapping_file_variable if provided as arguments
if args.input_dir is not None:
    config_dict['DATA_DIR']['INPUT_DIR'] = args.input_dir

if args.mapping_file is not None:
    config_dict['MAPPING_FILE'] =  args.mapping_file


# Save the updated config dictionary to the file
with open(config_path, 'w') as f:
    yaml.dump(config_dict, f)

# # load the configuration
config = config('/mnt/batch/tasks/shared/LS_root/mounts/clusters/optimus-aml-gpu/code/Users/ashwini_kumar/optimus_training_pipeline/data_transformer_training_pipeline/configs/config.yaml')

print (config)
# Check if the repository already exists
if os.path.exists(os.path.join(config.OUTPUT_DIR, config.TASK_ID)):
    print("Data Directory exist")
else:
    # Create the repository
    os.makedirs(os.path.join(config.OUTPUT_DIR, config.TASK_ID))
    print("Data Directory created..")

# Preprocess the data
out = create_new_cv_data(config.INPUT_DIR, config.CV_LIST, nsample=config.NSAMPLE)
# Add the coltypes and num classes on the go in config class so it can be used anywhere
coltypes,num_classes = extract_class(out[0])

config.COLTYPES = coltypes
config.NUM_CLASSES = num_classes

print (" Number of classes for training are :", config.NUM_CLASSES)
print (" The classes are as follows :",config.COLTYPES)

# Load the text file
with open('./models/doduo/model_inputs.txt', 'r') as f:
     lines = f.readlines()

# # Loop over the lines in the file and update the variables as needed
for i, line in enumerate(lines):
     if line.startswith('sato_coltypes'):
          lines[i] = f'sato_coltypes: {coltypes}\n'
     elif line.startswith('num_classes'):
          lines[i] = f'num_classes: {num_classes}\n'


# # Save the updated text file
with open('./models/doduo/model_inputs.txt', 'w') as f:
     f.writelines(lines)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")


# import subprocess

# # Define the command to run the script
# command_train = "python models/doduo/train_multi.py"
# train_script_path = 'models/doduo/test.py'
# subprocess.run(["python",train_script_path])

# Run the command
print ("Running Training")
train_script_path = 'models/doduo/train_multi.py'
subprocess.run(["python",train_script_path])
print ("Train Completed")

# Run the command
print ("Running Prediction")
predict_script_path = 'models/doduo/predict_multi.py'
subprocess.run(["python",predict_script_path])
print ("Train Completed")

# # Define the command to run the script
# command_test = "python models/doduo/predict_multi.py"

# Run the command
# print ("Running Testing")
# get_ipython().run_line_magic('run', 'models/doduo/predict_multi.py')
# print ("Testing Completed")
