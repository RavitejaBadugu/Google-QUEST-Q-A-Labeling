from fastapi import APIRouter,status
from pydantic import BaseModel
import pandas as pd
import numpy as np
from api_file.inputs_process import get_final_model_inputs
from utils.config_reader import inference_parameters
from api_file.load_model import make_predictions
from api_file.databasemodels import *
from datetime import datetime
from pydantic import create_model
from utils.y_labels import y_columns

router=APIRouter(prefix='/quality',tags=['main'])


class DATA(BaseModel):
    TITLE: str
    QUESTION: str
    ANSWER: str

Resp_model=create_model('RESPONSE_MODEL',**dict((col,0.001) for col in y_columns))


@router.post('/',status_code=status.HTTP_200_OK,response_model=Resp_model)
async def post_data(input_data: DATA):
    print('we got the user entered data')
    data=pd.DataFrame({'question_title': input_data.TITLE,
          'question_body': input_data.QUESTION,
           'answer': input_data.ANSWER},index=[0])
    print('features are yet to be added to database')
    add_features(input_data.QUESTION,input_data.ANSWER,
                input_data.TITLE,datetime.now())
    print('features are added to database')
    model_inputs=get_final_model_inputs(HEADS=inference_parameters['HEADS'],
                            PRE_NAME=inference_parameters['PRE_NAME'],
                            MAX_LENGTH=inference_parameters['MAX_LENGTH'],
                            tokenizer_path=inference_parameters['tokenizer_path'],
                            data=data,h1=inference_parameters['H1'],h2=inference_parameters['H2'],
                            inference=True)
    print('inputs to models are prepared')
    final_inputs=[{'input_1':np.squeeze(model_inputs['input_ids']).tolist(),
                    'input_2':np.squeeze(model_inputs['attention_mask']).tolist(),
                    'input_3':np.squeeze(model_inputs['token_type_ids']).tolist()}]
    predictions=make_predictions(final_inputs)
    final_predictions=dict((col,predictions[0,i]) for i,col in enumerate(y_columns))
    return final_predictions