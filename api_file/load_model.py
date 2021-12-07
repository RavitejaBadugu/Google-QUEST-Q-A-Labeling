from model_reader import parameters
import json

def make_predictions(model_inputs):
    model1=f"http://{parameters['HOST']}:{parameters['PORT']}/{parameters['VERSION']}/models/{parameters['MODEL_NAMES'][0]}"
    model2=f"http://{parameters['HOST']}:{parameters['PORT']}/{parameters['VERSION']}/models/{parameters['MODEL_NAMES'][1]}"
    model3=f"http://{parameters['HOST']}:{parameters['PORT']}/{parameters['VERSION']}/models/{parameters['MODEL_NAMES'][2]}"
    model4=f"http://{parameters['HOST']}:{parameters['PORT']}/{parameters['VERSION']}/models/{parameters['MODEL_NAMES'][3]}"
    model5=f"http://{parameters['HOST']}:{parameters['PORT']}/{parameters['VERSION']}/models/{parameters['MODEL_NAMES'][4]}"
    p1=json.loads(model1.predict(model_inputs)['prediction'])[0]
    p2=json.loads(model2.predict(model_inputs)['prediction'])[0]
    p3=json.loads(model3.predict(model_inputs)['prediction'])[0]
    p4=json.loads(model4.predict(model_inputs)['prediction'])[0]
    p5=json.loads(model5.predict(model_inputs)['prediction'])[0]
    p=(p1+p2+p3+p4+p5)/5
    return p