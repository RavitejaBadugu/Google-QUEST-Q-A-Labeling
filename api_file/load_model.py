from utils.config_reader import parameters
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
    p1=json.loads(requests.post(model1,data,headers=headers).text)
    p2=json.loads(requests.post(model2,data,headers=headers).text)
    p3=json.loads(requests.post(model3,data,headers=headers).text)
    p4=json.loads(requests.post(model4,data,headers=headers).text)
    p5=json.loads(requests.post(model5,data,headers=headers).text)
    p1=np.array(p1['predictions'])
    p2=np.array(p2['predictions'])
    p3=np.array(p3['predictions'])
    p4=np.array(p4['predictions'])
    p5=np.array(p5['predictions'])
    p=(p1+p2+p3+p4+p5)/5
    return p