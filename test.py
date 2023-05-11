#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %%
# %%
import pandas as pd
import os
import glob
import numpy as np
from visions import typesets
import random
import json
from sklearn.model_selection import train_test_split
from itertools import islice, cycle
import warnings
from itertools import cycle, takewhile, dropwhile
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import multiprocessing as mp
import tqdm
import itertools
warnings.filterwarnings("ignore")
import logging
import sys


# In[29]:


# %%
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('data_gen_logs_experiment.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

# %%
# %%
# training paths
# training paths
# training paths
input_data_folder_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/hitesh-cpu-8v/code/Users/hitesh_mehra/optimus/data_generation/filtered_train_test_data_partner/train_test_split/train"
print(input_data_folder_path)

input_mapping_csv_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/hitesh-cpu-8v/code/Users/hitesh_mehra/optimus/data_generation/util_files/fhir_mapping_loop_mimic_nhi_ecdi.csv"
saving_folder_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/hitesh-cpu-8v/code/Users/hitesh_mehra/optimus/data_generation/training_datasets/v4_train_data_mimic_loop_nhi_ecdi_tabert_included/output_generated_data"
print(saving_folder_path)

selab_saving_folder_path = os.path.join(saving_folder_path,"selab")
doduo_wide_saving_folder_path = os.path.join(saving_folder_path,"doduo_4096")
doduo_small_saving_folder_path = os.path.join(saving_folder_path,"doduo_512")
#tabert_saving_folder_path = os.path.join(saving_folder_path,"tabert")

os.makedirs(saving_folder_path, exist_ok=True)
os.makedirs(selab_saving_folder_path, exist_ok=True)
os.makedirs(doduo_wide_saving_folder_path, exist_ok=True)
os.makedirs(doduo_small_saving_folder_path, exist_ok=True)
#os.makedirs(tabert_saving_folder_path, exist_ok=True)

# list directories
input_files_list = os.listdir(input_data_folder_path)
print(input_files_list)

# mapping dict
data_read_dict = {"cci_stateview" : "file_read", "ECDI" : "file_read", "Loop" : "folder_read", "mimic_csv" : "file_read"}
data_count_dict = {"cci_stateview" : 5000, "ECDI" : 7000, "Loop" : 3000, "mimic_csv" : 3000}

print(data_read_dict)

# %%
# %%


# In[6]:


def file_list_generator():

    all_files = []
    for folder_name in input_files_list:
        #print(folder_name)
        if folder_name in data_read_dict.keys():
            
            if data_read_dict[folder_name] == "file_read":
                file_list = glob.glob(os.path.join(input_data_folder_path,folder_name,"*.csv"))
                file_list1 = glob.glob(os.path.join(input_data_folder_path,folder_name,"*.parquet"))
                total_list = file_list + file_list1
                random.shuffle(total_list)
                
                
                for x in total_list:
                    all_files.append([x,x.split("/")[-1][:-4].lower(),folder_name])

            else:
                folder_list = os.listdir(os.path.join(input_data_folder_path,folder_name))
                for i  in folder_list:
                    #print(i)
                    file_paths_csv = glob.glob(os.path.join(input_data_folder_path,folder_name,i,"*.csv"))
                    file_paths_csv1 = glob.glob(os.path.join(input_data_folder_path,folder_name,i,"*","*.csv"))
                    file_paths_parquet = glob.glob(os.path.join(input_data_folder_path,folder_name,i,"*.parquet"))
                    file_paths_parquet1 = glob.glob(os.path.join(input_data_folder_path,folder_name,i,"*","*.parquet"))
                    file_paths_parquet_combined = file_paths_parquet + file_paths_parquet1
                    
                    random.shuffle(file_paths_parquet_combined)
                    
                    for ind ,f in enumerate(file_paths_parquet_combined):
                        temp = pd.read_parquet(f)
                        if temp.shape[0] >= 500:
                            file_paths_parquet_combined = [file_paths_parquet_combined[ind]]
                            #print(file_paths_parquet_combined, "434424" )       
                            break
                        
                        else:
                            continue         
                    
                    
                    
                    
                    filename_list = file_paths_csv  +file_paths_csv1 + file_paths_parquet_combined
                    if len(filename_list)>0:

                        random.shuffle(filename_list)
                        all_files.append([filename_list[0],i.lower(),folder_name ])


    #all_files = list(itertools.chain.from_iterable(all_files))     
    
    return all_files
# %%
# generate dataframe containing table name and column names

def data_column_generator(input_folder_path,filename, name_id = None):
    
    
    name, extension = os.path.splitext(filename)
    
    if name_id : 
        name = name_id
    if extension == ".csv":
        df = pd.read_csv(os.path.join(input_folder_path,filename))
    else:
        df = pd.read_parquet(os.path.join(input_folder_path,filename))
    #print(df)
    return pd.DataFrame({"Table_Name" : [name.lower()] * len(df.columns), "Column_Name" : map(str.lower,df.columns)})





def data_generator_mode1(folder_name):
    
    folder_list = os.listdir(os.path.join(input_data_folder_path,folder_name))
    print(folder_list)
    col_list = []
    for i in folder_list:
        #print(i)
        
        file_paths_csv = glob.glob(os.path.join(input_data_folder_path,folder_name,i,"*.csv"))
        file_paths_csv1 = glob.glob(os.path.join(input_data_folder_path,folder_name,i,"*","*.csv"))
        file_paths_parquet = glob.glob(os.path.join(input_data_folder_path,folder_name,i,"*.parquet"))
        file_paths_parquet1 = glob.glob(os.path.join(input_data_folder_path,folder_name,i,"*","*.parquet"))
        
        filename_list = file_paths_csv + file_paths_parquet +file_paths_csv1 + file_paths_parquet1
        #print(filename_list)
        if len(filename_list)>0:
            
            filename = filename_list[0]
            name, extension = os.path.splitext(filename)
            
            name = i
            
            if extension == ".csv":
                df = pd.read_csv(filename, nrows=2)
                

            else:
                df = pd.read_parquet(filename)
                
            col_list.append(pd.DataFrame({"Table_Name" : [name.lower()] * len(df.columns), "Column_Name" : map(str.lower,df.columns)}))

            
    
    return pd.concat(col_list, axis=0)
        
    
    
    
def data_generator_mode2(folder_name):
    
    '''
    for cases for folder like mimic and cci, which are in structure table_name.csv
    '''
    col_list = list()
    input_folder_path = os.path.join(input_data_folder_path,folder_name)
    files_list = os.listdir(input_folder_path)
    for filename in files_list:
        
        
        name, extension = os.path.splitext(filename)
        if extension in [".csv", ".parquet"]:
            
    
            if extension == ".csv":
                df = pd.read_csv(os.path.join(input_folder_path,filename), nrows=2)
                

            else:
                df = pd.read_parquet(os.path.join(input_folder_path,filename))
                
            col_list.append(pd.DataFrame({"Table_Name" : [name.lower()] * len(df.columns), "Column_Name" : map(str.lower,df.columns)}))

        else:
            continue
            
        
        #print(df)
    return pd.concat( col_list, axis=0)

        
    #with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        #columns =  pd.concat(list(executor.map(data_column_generator,file_path_list )), axis=0)
    
    


# In[7]:


# %%
# %%
# get input_data
col_list = []
for i in tqdm.tqdm(input_files_list):
    #print(i)
    if i in data_read_dict.keys():
        model_method = data_read_dict[i]
        if model_method == "file_read":
            
            col_list.append(data_generator_mode2(i))
            
            
        else:
            col_list.append(data_generator_mode1(i))
            
            
    else:
        continue

columns = pd.concat(col_list, axis=0)
    
  


  

# %%
columns


# In[8]:


# %%
# %%
# # seperated sheets provided for facility and physician

# to be discussed for mappings without labels and unsupported
fhir_mapping = pd.read_csv(input_mapping_csv_path)
#mapping_excel = pd.ExcelFile(input_mapping_csv_path)
#mapping_list = [mapping_excel.parse(os.path.splitext(x)[0].lower()).loc[:,["Table_Name","Column_Name","Target_Mapping"]] for x in input_files]
#fhir_mapping = pd.concat(mapping_list, axis=0)
#fhir_mapping["Target_Mapping"] = fhir_mapping["Target_Mapping"].fillna("not supported")



fhir_mapping = fhir_mapping.apply(lambda x : x.astype("str").str.lower())
fhir_mapping.columns= [i.strip() for i in fhir_mapping.columns]
fhir_mapping.Table_Name = fhir_mapping.Table_Name.str.strip()
fhir_mapping.Column_Name = fhir_mapping.Column_Name.str.strip()

# Inner Join the fhir mapping columns with columns dataframe created. Only those headers which have absolute FHIR mapping are considered 
fhir_table_mapping = columns.merge(fhir_mapping, how = 'inner', left_on = ['Table_Name','Column_Name'], right_on = ['Table_Name','Column_Name'])
fhir_table_mapping.index = fhir_table_mapping.Table_Name + '_' + fhir_table_mapping.Column_Name

# %%
# %%
fhir_table_mapping.head()

# %%
# %%
len(fhir_table_mapping)


# In[9]:


# %%
# %%
# geenrate mapping json

# dictionary creation
# This code is commented out as we want consitency so we always reads from csv & convert to dict
# # Select all the unique FHIR mapping from the table fhir table mapping
fhir_col = list(fhir_table_mapping['Target_Mapping'].unique())
# # Convert them to lower case & select unqiue (some fhir mapping has variations on case)
fhir_col_type= list(set([x.lower() for x in fhir_col]))
# # Create a dictionary with FHIR mapping and assigna sequential id to them This will be used as labels while training
#fhir_coltype_id_dict = dict([(name, i) for i, name in enumerate(fhir_col_type)])

fhir_coltype_id_dict= {'condition.encounter': 0, 'procedure.code.coding.system': 1, 'medicationadministration.context': 2, 'condition.code.coding.display': 3, 'others': 4, 'observation.valuequantity.unit': 5, 'organization.contact.telecom.phone': 6, 'relatedperson.name.text': 7, 'observation.category.coding.aloholusersnomed': 8, 'observation.note': 9, 'organization.contact.address.postalcode': 10, 'proceudre.performedperiod.start': 11, 'medicationrequest.dosageinstruction.timing.code': 12, 'observation.referencerange.low': 13, 'medicationadministration.subject': 14, 'medicationrequest.dosageinstructions.doseandrate.ratequantity': 15, 'medicationadministration.request': 16, 'patient.contact.address.postalcode': 17, 'specimen.type': 18, 'observation.category.coding.smokingusersnomed': 19, 'patient.contact.telecon.home': 20, 'medicationdispense.identifier': 21, 'observation.valuedatetime': 22, 'observation.extension.value': 23, 'observation.extension.lab_priority': 24, 'medicationknowledge.synonym': 25, 'observation.extension.comparator': 26, 'patient.gender': 27, 'procedure.code.coding.display': 28, 'relatedperson.contact.telecom.work': 29, 'coverage.period.end': 30, 'observation.category.coding.aloholcommentsnomed': 31, 'observation.subject': 32, 'encounter.status': 33, 'communication.language': 34, 'patient.birthdate': 35, 'observation.status': 36, 'encounter.reasonreference.observation.code.coding_body_mass_index': 37, 'patient.contact.address.line': 38, 'questionnaire.item.answeroption.value.valuedate': 39, 'medicationrequest.identifier': 40, 'patient.contact.address.city': 41, 'medicationrequest.dosageinstruction.route': 42, 'specimen.type.code': 43, 'coverage.subscriberid': 44, 'coverage.period.start': 45, 'encounter.reasonreference.observation.code.coding_height': 46, 'coverage.payor.organization.id': 47, 'specimen.type.display': 48, 'medicationadministration.effectiveperiod.end': 49, 'observation.code.display': 50, 'encounter.period.value': 51, 'encounter.participant.individual.practitioner.qualification': 52, 'procedure.status': 53, 'condition.code.coding.system': 54, 'encounter.reasonreference.observation.identifier': 55, 'procedure.performeddatetime': 56, 'familymemberhistory.relationship': 57, 'patient.contact.telecon.work': 58, 'medicationadministration.dosage.dose.unit': 59, 'observation.component.code.text': 60, 'medicationrequest.dosageinstructions.doseandrate.dosequantity.unit': 61, 'encounter.reasonreference.observation.component.code_weight': 62, 'encounter.period': 63, 'medicationadministration.effectiveperiod.start': 64, 'observation.encounter.reference': 65, 'medicationadministration.dosage.ratequantity.value': 66, 'medicationdispense.encounter': 67, 'organization.contact.name': 68, 'observation.referencerange.high': 69, 'organization.type.text': 70, 'condition.onsetdatetime': 71, 'observationeffectivedatetime': 72, 'procedure.identifier': 73, 'observation.encounter': 74, 'encounter.identifier': 75, 'patient.contact.telecom.phone': 76, 'organization.contact.address.city': 77, 'condition.identifier': 78, 'procedure.encounter': 79, 'encounter.reasonreference.observation.code.coding_temperature': 80, 'condition.code.coding.code': 81, 'organization.contact.address.line': 82, 'procedure.code.coding.code': 83, 'observation.subject.reference': 84, 'medicationrequest.dosageinstructions.doseandrate.dosequantity.value': 85, 'organization.contact.name.given': 86, 'specimen.identifier': 87, 'encounter.partof': 88, 'patient.deceasedboolean': 89, 'patient.identifier': 90, 'observation.valuequantity.value': 91, 'encounter.serviceprovider.organization.identifier': 92, 'medicationrequest.status': 93, 'observation.referencerange.normalvalue': 94, 'observation.issued': 95, 'medicationdispense.quantity': 96, 'proceudre.code.coding.code': 97, 'medicationadministration.dosage.dose.value': 98, 'medicationadministration.effectivedatetime': 99, 'procedure.performedperiod.end': 100, 'patient.contact.name.suffix': 101, 'medicationrequest.dispenserequest.validityperiod.end': 102, 'encounter.participant.individual.practitioner.telecom.fax': 103, 'coverage.payor.organization.name': 104, 'patient.contact.name.given': 105, 'medicationrequest.dispenserequest.numberofrepeatsallowed': 106, 'observation.interpretation': 107, 'observation.valuecodeableconcept': 108, 'medicationrequest.dosageinstruction.timing.repeat.durationunit': 109, 'medicationrequest.dosageinstruction.text': 110, 'relatedperson.contact.telecom.mobile': 111, 'encounter.hospitalization.dischargedisposition': 112, 'medicationadministration.status': 113, 'medicationrequest.dispenserequest.validityperiod.start': 114, 'relatedperson.contact.telecom.home': 115, 'medicationadministration.category': 116, 'encounter.subject': 117, 'observation.category': 118, 'observation.code.coding.display': 119, 'medication.form': 120, 'medicationrequest.subject': 121, 'practitionerrole.specialty': 122, 'observation.code.code': 123, 'extension.ethnicity': 124, 'observation.code.coding.code': 125, 'condition.subject': 126, 'patient.contact.address.state': 127, 'patient.deceaseddatetime': 128, 'encounter.type': 129, 'medicationrequest.dosageinstruction.doseandrate.ratequantity': 130, 'medicationrequest.courseoftherapytype': 131, 'organization.contact.telecom.fax': 132, 'medicationrequest.medicationreference': 133, 'organization.contact.address.state': 134, 'medicationadministration.dosage.route': 135, 'medicationadministration.dosage.site': 136, 'encounter.reasonreference.observation.code_bloodpressure': 137, 'procedure.code.text': 138, 'encounter.participant.individual.practitioner.telecom.phone': 139, 'medication.code': 140, 'encounter.period.end': 141, 'observation.identifier': 142, 'observation.valuestring': 143, 'questionnaire.identifier': 144, 'encounter.hospitalization.admitsource': 145, 'medicationadministration.identifier': 146, 'procedure.bodysite': 147, 'maritalstatus': 148, 'procedure.subject': 149, 'organization.identifier': 150, 'medicationrequest.encounter': 151, 'encounter.participant.individual.practitioner.name': 152, 'medicationdispense.subject': 153, 'questionnaire.name': 154, 'encounter.reasonreference.observation.code.coding_pulse': 155, 'medicationrequest.dosageinstruction.timing.maxdoseperperiod': 156, 'condition.abatementdatetime': 157, 'encounter.reasoncode.text': 158, 'observation.effectivedatetime': 159, 'medicationadministration.dosage.ratequantity.rateuom': 160, 'questionnaire.item.answeroption.value.valuestring': 161, 'medicationrequest.dosageinstruction.timing.repeat.duration': 162, 'medicationadministration.dosage.ratequantity.unit': 163, 
'medicationrequest.authoredon': 164, 'encounter.period.start': 165}
#with open(os.path.join(saving_folder_path,'label_mapping.json'), 'w') as fp:
#    json.dump(fhir_coltype_id_dict, fp)

        
    
    
current_count = len(fhir_coltype_id_dict)
new_keys= set(fhir_col_type).difference(fhir_coltype_id_dict.keys())
for i in new_keys:
    fhir_coltype_id_dict[i] = current_count
    current_count = current_count+1
#with open(os.path.join(saving_folder_path,'label_mapping.json'), 'w') as fp:
#    json.dump(fhir_coltype_id_dict, fp)

with open("util_files/loop_mimic_ecdi_nhi_label_mapping.json") as user_file:
     fhir_coltype_id_dict = json.load(user_file)
    

# %%
print(len(fhir_coltype_id_dict))


# In[10]:


# %%
# %%

class CyclicalList:
    def __init__(self, initial_list):
        self._initial_list = initial_list

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.stop is None:
                raise ValueError("Cannot slice without stop")
            iterable = enumerate(cycle(self._initial_list))
            if item.start:
                iterable = dropwhile(lambda x: x[0] < item.start, iterable)
            return [
                element
                for _, element in takewhile(lambda x: x[0] < item.stop, iterable)
            ]

        for index, element in enumerate(cycle(self._initial_list)):
            if index == item:
                return element

    def __iter__(self):
        return cycle(self._initial_list)


# In[11]:


# %%
def sliding_window(iterable,size, overlap= 0):
    start = 0
    end = size
    step = size - overlap
    length = len(iterable)
    while end < length:
        yield tuple(iterable[start:end])
        start += step
        end += step
    yield tuple(iterable[start:])

# %%
# table df for count

table_df = pd.DataFrame([[0,49,5,5],
    [50,99,8,8],
    [100,999999999999,20,10]])
table_df.columns = ["min_col_count","max_col_count","strategy3_iteration_count","strategy4_iteration_count"]


# %%


# In[ ]:





# In[12]:


# %%
def table_generation_strategy(df):
    
    output_df = []
    
    if df.shape[1] <=20:
        output_df.append(df)
        return output_df
    
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
    
    if df.shape[1] > 19:
        
        # strategy2 :  sliding window of 20 with overlap of 10

        sliding_window_list = list(sliding_window(range(df.shape[1]), 20,10))
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



        # strategy 4 :  generate 10 sets of randomly selected  20 columns
        if df.shape[1] <=20:
                sampler_count = df.shape[1]-2

        else:
            sampler_count = 20






        for i in range(strategy4_iteration_count):
            generated_indice = random.sample(range(df.shape[1]),sampler_count)
            output_df.append(df.iloc[:,generated_indice])

    
    return output_df


# In[13]:


# %%
def table_augmentation(df_part):
    generate_rand = random.randint(0,10)
    indices = 0 
    
    # if random number is 0 or divisible by 7 then delete one column at random
    if generate_rand % 7 == 0 or generate_rand % 8 == 0:
        df_part  = df_part.drop(df_part.columns[[random.randint(0,df_part.shape[1]-1)]],axis = 1)
    
    # If random number is 0 or divisible by 6 then shuffle columns
    if generate_rand % 6 == 0 or generate_rand % 9 == 0:
        column_list = list(df_part.columns.values)
        random.shuffle(column_list)
        # print (column_list)
        df_part = df_part[column_list]
        
    
    return df_part

def split_dataframe(df):
    number_of_sub_dataframes = len(df)//16
    return [df.iloc[i:i + 16,:] for i in range(0, len(df), 16)][:number_of_sub_dataframes]

# %%


# In[14]:


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



null_removal_table_df = pd.DataFrame([[0,1000,0.9],
    [1001,10000,0.95],
    [10001,100000,0.99],
    [100001,100000000,1,]])
null_removal_table_df.columns = ["min_table_length", "max_table_length","remove_threshold"]   

# remove null data
def get_null_removed_data(temp_df):
    
        count = 0
        row_count = len(temp_df)
        print(row_count)
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



# %%
# %%


# In[ ]:



    
    
    
    
    


# In[30]:


def create_new_cv_data(cv_count):
    tabert_df_list = list()
    validation_list = list()
    
    nsample = 3000
    for cv in tqdm.tqdm(cv_count):
        selab_cv_folder_path = os.path.join(selab_saving_folder_path, str(cv))
        os.makedirs(selab_cv_folder_path, exist_ok=True)
        #tabert_cv_folder_path = os.path.join(tabert_saving_folder_path, str(cv))
        #os.makedirs(tabert_cv_folder_path, exist_ok=True)
        
        print ("Create data for the cv :",cv)
        logger.info(f"Create data for the cv : {cv}")
        # Create a empty column list
        col_list = []
        # Create a empty data list
        data_list_wide = []
        data_list_512 = []
        # Run inner for loop for 700 time, can be choosed individually
        for iteration in range(1):

            i = 0
            k = 0
            indices = 0
            df_iden = 0
            # Logic to do for mimic
            all_files = file_list_generator()
            random.shuffle(all_files) 
            #print(all_files)
            
            
            for file_ind,current_file in enumerate(all_files):
                data_count = []
                
                try:
                    
                    file = current_file[0]
                    table_name = current_file[1]
                    source_name = current_file[2]
                    print(file,table_name,source_name)
                    sample_count = data_count_dict[source_name]
                    nsample = data_count_dict[source_name]
                    
                    data_count.extend([file, table_name,source_name])
                    
                    # loading csv files
                    if file.endswith((".csv", ".parquet")):
                        
                        #filename = file.split("/")[-1]
                        #table_name = filename[:-4].lower()

                        #if table_name in table_dict.keys():
                    
                        table_id_temp = table_name + "_cv" + str(cv) 

                        # os.path.join(csvpaths, file)
                        
                        
                        logger.info(f"Started reading table : {table_name} from cv : {cv}, having table index: {file_ind} out of {len(all_files)}")
                        
                        
                        
                        

                        if file in ['admissions.csv','chartevents.csv','datetimeevents.csv','diagnoses_icd.csv','emar.csv','emar_detail.csv','inputevents.csv','labevents.csv','microbiologyevents.csv','outputevents.csv','patients.csv','pharmacy.csv','poe.csv','prescriptions.csv','procedureevents.csv','procedures_icd.csv']:
                                                    # Read the selected rows from the CSV file
                            #print ("Big file processed in if :",file)
                            
                                
                            df_temp = pd.read_csv( file, nrows = 299000)    
                            df = df_temp.sample(nsample)
                            
                            
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
                        
                        df = df.loc[:,df.columns.str.lower().isin(fhir_table_mapping[fhir_table_mapping['Table_Name'] == table_name ]["Column_Name"].tolist())]
                        
                        print("mapped_data_count", df.shape[1])
                        data_count.append(df.shape[1])
                        #print ("After",df.shape)  
                        
                        # add null and stratified sampling
                        # remove null columns
                        
                        
                        
                        # generate tabert df
                        '''
                        if df.shape[1] > 0:
                            
                            tabert_columns = [[table_name,x.lower(),fhir_table_mapping.loc[ table_name + '_' + x.lower(),"Target_Mapping"]] for x in df.columns]
                            data_count.append(len(tabert_columns))   
                            tabert_df = pd.DataFrame(tabert_columns)
                        
                        else:
                            tabert_df = pd.DataFrame(columns = range(3))
                            data_count.append(0)
                        tabert_df.columns = ["table_name", "original_table_name","fhir_mapping_target_name"]
                        df.to_csv(os.path.join(tabert_cv_folder_path,table_name + ".csv"), index=False)

                        
                        '''
                        
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
                        
                        
                        #print(df.isna().isnull().any(), "sampler_output")
                        #print(len(df), "sampler_output_df")
                        #print(df.columns, "filtered_columns")
                        




                        # generate strategic tables based on four strategies :  output 40 tables

                        output_tables = table_generation_strategy(stratified_df)
                        print("stratgeized_table_count", len(output_tables))
                        data_count.append(len(output_tables))
                        validation_list.append(data_count)
                        
                        

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
        
                      
                        

# %%
    

# %%

if __name__=="__main__":

    k = create_new_cv_data([0])
    

# %%


# In[35]:


#cm = a.values
##tp = np.diag(cm)
#prec = list(map(truediv, tp, np.sum(cm, axis=0)))
#rec = list(map(truediv, tp, np.sum(cm, axis=1)))
#print ('Precision: {}\nRecall: {}'.format(prec, rec))

