

curl -X POST -H "Content-Type: application/json" -d '{"raw_data_directory": "/mnt/batch/tasks/shared/LS_root/mounts/clusters/optimus-aml-gpu/code/Users/ashwini_kumar/Optimus_Data_Transformer_pipeline/Data/input_data/"}' http://localhost:8000/transform_data



curl -X POST -F "raw_data_directory=/mnt/batch/tasks/shared/LS_root/mounts/clusters/optimus-aml-gpu/code/Users/ashwini_kumar/Optimus_Data_Transformer_pipeline/Data/input_data/" http://localhost:8000/transform_data


curl -X POST "http://localhost:8000/transform_data?raw_data_directory=/mnt/batch/tasks/shared/LS_root/mounts/clusters/optimus-aml-gpu/code/Users/ashwini_kumar/Optimus_Data_Transformer_pipeline/Data/input_data" -H "accept: application/json"

Install FastAPI: pip install fastapi

Install uvicorn server: pip install uvicorn