import os
import shutil
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from data_transformer import data_transformer_prediction


app = FastAPI()

def data_transformer_list_parser(raw_data_directory):
    
    global doduo_embeddings_list 
    doduo_embeddings_list = list()
    
    # location for storing intermediate outputs from data transformer
    intermediate_data_location = "intermediate_output"
    os.makedirs(intermediate_data_location, exist_ok=True)

    canonical_save_location  = os.path.join(intermediate_data_location,"canonical_input_data")

    if os.path.exists(canonical_save_location):
        shutil.rmtree(canonical_save_location)

    os.makedirs(canonical_save_location,exist_ok=True)

    # initiate data transformer class
    data_transformer = data_transformer_prediction()
    raw_data_file_list = [x.filename for x in raw_data_directory]

    for file_name in raw_data_file_list:
        table_name = file_name.split(".csv")[0]
        input_df = pd.read_csv(file_name)
        canonical_data_location = os.path.join(canonical_save_location, file_name.split(".csv")[0])
        os.makedirs(canonical_data_location, exist_ok=True)

        # data transformer output
        output = data_transformer.input_table_annotations(input_df, table_name)

        # store mapping 
        output[0].to_csv(os.path.join(canonical_data_location, "data_mapping.csv"), index=False)

        # store stratified data
        output[1].to_csv(os.path.join(canonical_data_location, "stratified_data.csv"), index=False)

        # store pickle file for doduo embeddings on the data
        doduo_embeddings_list.append(output[2])

    doduo_embeddings_df = pd.concat(doduo_embeddings_list, axis=0)
    doduo_embeddings_df.reset_index(drop=True, inplace=True)
    doduo_embeddings_df.to_pickle(os.path.join(canonical_save_location, "data_transformer_embeddings.pkl"))
    del doduo_embeddings_list 


@app.post("/data_transformer")
async def transform_data(raw_data_directory: UploadFile):
    with open(raw_data_directory, "r") as file:
        data = file.readlines()

    # write data to csv file
    with open("input_data.csv", "w") as f:
        f.writelines(data)

    raw_data_directory = "input_data.csv"
    data_transformer_list_parser(raw_data_directory)

    return {"message": "Data transformation complete!"}



In this modified code, we added a new /data_transformer endpoint to the FastAPI app. This endpoint accepts a file upload
 and writes its content to a CSV file named input_data.csv. We then call the data_transformer_list_parser function with the file path as input to process the data
 and perform the necessary transformations. Finally, we return a JSON response indicating that the transformation is complete.
 
 
 To test the FastAPI app using curl, you can follow these steps:

Start the FastAPI app by running the Python script containing the modified code.
In a separate terminal window, navigate to the directory containing the CSV file you want to upload for testing.
Use the following curl command to upload the file and test the /data_transformer endpoint:
bash
Copy code
curl -X POST -F "raw_data_directory=@<path-to-file>" http://localhost:8000/data_transformer
Replace <path-to-file> with the actual path to your CSV file.

After running the curl command, you should see a response indicating that the data transformation is complete.
Note: Make sure to replace http://localhost:8000 with the appropriate URL if you're running the app on a different host or port.






Regenerate respo