import yaml

with open('config.yml') as f:
    parameters=yaml.safe_load(f)

inference_parameters=parameters['INFERENCE_PARAMETERS']