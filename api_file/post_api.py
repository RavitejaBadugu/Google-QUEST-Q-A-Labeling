from fastapi import APIRouter,HTTPException
from pydantic import BaseModel
import pandas as pd
from tensorflow.keras import models
from api_file.inputs_process import get_final_model_inputs
from api_file.config_reader import inference_parameters,model_paras
from api_file.load_model import MODELS
from api_file.model_file import SINGLE_MODEL
router=APIRouter(prefix='/quality',tags=['main'])


class DATA(BaseModel):
    TITLE: str
    QUESTION: str
    ANSWER: str

print(model_paras)
models=MODELS(SINGLE_MODEL(PRE_NAME=inference_parameters['PRE_NAME'],
                                  MAX_LENGTH=inference_parameters['MAX_LENGTH'],
                                  PRE_MODEL=model_paras['pre_trained_model'],
                                  sequence=False,
                                  final_activation=True,
                                  hidden_states=True, 
                                  hidden_number=4),model_paras['fold_models'])

@router.post('/')
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
    predictions=models.predict(model_inputs)
    print(predictions)