from fastapi import APIRouter,status
from pydantic import BaseModel
import pandas as pd
import numpy as np
from api_file.inputs_process import get_final_model_inputs
from api_file.config_reader import inference_parameters
from api_file.load_model import make_predictions

router=APIRouter(prefix='/quality',tags=['main'])


class DATA(BaseModel):
    TITLE: str
    QUESTION: str
    ANSWER: str

class Output_model(BaseModel):
    pass

@router.post('/',status_code=status.HTTP_200_OK)
async def post_data(input_data: DATA):
    data=pd.DataFrame({'question_title': input_data.TITLE,
          'question_body': input_data.QUESTION,
           'answer': input_data.ANSWER},index=[0])
           
    model_inputs=get_final_model_inputs(HEADS=inference_parameters['HEADS'],
                            PRE_NAME=inference_parameters['PRE_NAME'],
                            MAX_LENGTH=inference_parameters['MAX_LENGTH'],
                            tokenizer_path=inference_parameters['tokenizer_path'],
                            data=data,h1=inference_parameters['H1'],h2=inference_parameters['H2'],
                            inference=True)
    final_inputs=[{'input_1':np.squeeze(model_inputs['input_ids']).tolist(),
                    'input_2':np.squeeze(model_inputs['attention_mask']).tolist(),
                    'input_3':np.squeeze(model_inputs['token_type_ids']).tolist()}]
    predictions=make_predictions(final_inputs).tolist()
    return predictions