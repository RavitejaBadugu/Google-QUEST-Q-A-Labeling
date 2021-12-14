from api_file.config_reader import parameters
import json
import requests
import numpy as np

def make_predictions(instances):
    model1=f"http://{parameters['HOST']}:{parameters['PORT']}/{parameters['VERSION']}/models/{parameters['MODEL_NAMES'][0]}:predict"
    model2=f"http://{parameters['HOST']}:{parameters['PORT']}/{parameters['VERSION']}/models/{parameters['MODEL_NAMES'][1]}:predict"
    model3=f"http://{parameters['HOST']}:{parameters['PORT']}/{parameters['VERSION']}/models/{parameters['MODEL_NAMES'][2]}:predict"
    model4=f"http://{parameters['HOST']}:{parameters['PORT']}/{parameters['VERSION']}/models/{parameters['MODEL_NAMES'][3]}:predict"
    model5=f"http://{parameters['HOST']}:{parameters['PORT']}/{parameters['VERSION']}/models/{parameters['MODEL_NAMES'][4]}:predict"
    data=json.dumps({"signature_name": "serving_default",'instances':instances})
    headers={"content-type": "application/json"}
    p1=np.array(json.loads(requests.post(model1,data,headers=headers).text)['predictions'])
    p2=np.array(json.loads(requests.post(model2,data,headers=headers).text)['predictions'])
    p3=np.array(json.loads(requests.post(model3,data,headers=headers).text)['predictions'])
    p4=np.array(json.loads(requests.post(model4,data,headers=headers).text)['predictions'])
    p5=np.array(json.loads(requests.post(model5,data,headers=headers).text)['predictions'])
    p=(p1+p2+p3+p4+p5)/5
    return p
