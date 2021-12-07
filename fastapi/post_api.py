from fastapi import APIRouter,HTTPException
from pydantic import BaseModel
from typing import Dict
import numpy as np
import pandas as pd
from inputs_process import get_final_model_inputs
from config_reader import inference_parameters

router=APIRouter(prefix='/quality',tags=['main'])


class DATA(BaseModel):
    TITLE: str
    QUESTION: str
    ANSWER: str

class Output_class(BaseModel):
    input_ids: np.array
    attention_mask: np.array
    token_type_ids: np.array

@router.post('/',response_model=Output_class)
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

