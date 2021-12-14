import yaml

with open('api_file/config.yml') as f:
    parameters=yaml.safe_load(f)

inference_parameters=parameters['INFERENCE_PARAMETERS']['general']

with open('api_file/database.yaml') as f:
    database_paras=yaml.safe_load(f)

with open('api_file/models.yaml') as f:
    parameters=yaml.safe_load(f)

