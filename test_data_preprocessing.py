#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import sys
import os
import shutil
import pandas as pd
import sys
import glob
import tqdm
sys.path.append('../')

from data_loader.data_preprocessing import *
from configs.config import config
import argparse

# load the configuration
config = config('/mnt/batch/tasks/shared/LS_root/mounts/clusters/optimus-aml-gpu/code/Users/ashwini_kumar/optimus_training_pipeline/data_transformer_training_pipeline/configs/config.yaml')



# Check if the repository already exists
if os.path.exists(os.path.join(config.OUTPUT_DIR, config.TASK_ID)):
    print("Data Directory exist")
else:
    # Create the repository
    os.makedirs(os.path.join(config.OUTPUT_DIR, config.TASK_ID))
    print("Data Directory created..")


# In[2]:


out = create_new_cv_data(config.INPUT_DIR, config.CV_LIST, nsample=config.NSAMPLE)


# In[3]:


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


# In[4]:


import torch
torch.cuda.is_available()


# In[5]:


import IProgress
import ipywidgets
get_ipython().run_line_magic('run', 'models/doduo/train_multi.py')


# In[7]:


get_ipython().run_line_magic('run', 'models/doduo/predict_multi.py')

import subprocess

# Define the command to run the script
command = "python models/doduo/predict_multi.py"

# Run the command
subprocess.run(command, shell=True)


import pickle

# ... Existing code ...

# Pickle the NMF model
with open('/content/drive/MyDrive/topic_modelling/results/nmf_model.pkl', 'wb') as file:
    pickle.dump(nmf, file)

# Pickle the TF-IDF vectorizer
with open('/content/drive/MyDrive/topic_modelling/results/tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)


import pickle

# Load the NMF model
with open('/content/drive/MyDrive/topic_modelling/results/nmf_model.pkl', 'rb') as file:
    nmf = pickle.load(file)

# Load the TF-IDF vectorizer
with open('/content/drive/MyDrive/topic_modelling/results/tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Example usage on new text data
new_text = "This is a new document to classify."

# Transform the new text using the TF-IDF vectorizer
new_text_tfidf = tfidf_vectorizer.transform([new_text])

# Get the topic distribution for the new text
new_text_topic_distribution = nmf.transform(new_text_tfidf)

# Find the top topic for the new text
new_text_top_topic = np.argmax(new_text_topic_distribution)

# Print the top topic number and corresponding words
print("Top Topic Number:", new_text_top_topic)
print("Top Topic Words:", ' '.join(get_top_words(nmf, tfidf_vectorizer.get_feature_names(), n_top_words=10)[new_text_top_topic]))
