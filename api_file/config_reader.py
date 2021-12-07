import yaml

with open('api_file/config.yml') as f:
    parameters=yaml.safe_load(f)

inference_parameters=parameters['INFERENCE_PARAMETERS']['general']
model_paras=parameters['INFERENCE_PARAMETERS']['models']