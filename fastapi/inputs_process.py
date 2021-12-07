import numpy as np
from process import *

def get_inputs(PRE_NAME,MAX_LENGTH,tokenizer_path,data,h1,h2,inference=False):
    columns=data.columns.tolist()
    y_columns=['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']
    if len(h2)==0:# single head model
        INPUT_IDS=np.empty((data.shape[0],MAX_LENGTH),dtype=np.int32)
        ATTENTION_MASK=np.empty((data.shape[0],MAX_LENGTH),dtype=np.int32)
        if not inference:
            Y=np.empty((data.shape[0],30))
        if not PRE_NAME.startswith('roberta'):
            TOKEN_TYPE_IDS=np.empty((data.shape[0],MAX_LENGTH),dtype=np.int32)
        for i in range(data.shape[0]):
            t1,t2=h1
            s1=' '.join(data.loc[i,t1].values)
            s2=' '.join(data.loc[i,t2].values)
            if PRE_NAME.startswith('bert'):
                INPUT_IDS[i,],ATTENTION_MASK[i,],TOKEN_TYPE_IDS[i,]=get_bert_single_head_inputs(Tokenizer(PRE_NAME,tokenizer_path),s1,s2,MAX_LENGTH)
            elif PRE_NAME.startswith('xlnet'):
                INPUT_IDS[i,],ATTENTION_MASK[i,],TOKEN_TYPE_IDS[i,]=get_xlnet_single_head_inputs(Tokenizer(PRE_NAME,tokenizer_path),s1,s2,MAX_LENGTH)
            else:
                INPUT_IDS[i,],ATTENTION_MASK[i,]=get_roberta_single_head_inputs(Tokenizer(PRE_NAME,tokenizer_path),' '+s1,' '+s2,MAX_LENGTH)
            if not inference:
                Y[i,]=data.loc[i,y_columns]
        if not PRE_NAME.startswith('roberta') and not inference:
            return INPUT_IDS,ATTENTION_MASK,TOKEN_TYPE_IDS,Y
        if not PRE_NAME.startswith('roberta') and inference:
            return INPUT_IDS,ATTENTION_MASK,TOKEN_TYPE_IDS
        else:
            if not inference:
                return INPUT_IDS,ATTENTION_MASK,Y
            else:
                return INPUT_IDS,ATTENTION_MASK
    else:
        INPUT_IDS1=np.empty((data.shape[0],MAX_LENGTH),dtype=np.int32)
        ATTENTION_MASK1=np.empty((data.shape[0],MAX_LENGTH),dtype=np.int32)
        INPUT_IDS2=np.empty((data.shape[0],MAX_LENGTH),dtype=np.int32)
        ATTENTION_MASK2=np.empty((data.shape[0],MAX_LENGTH),dtype=np.int32)
        if not inference:
            Y=np.empty((data.shape[0],30))
        if not PRE_NAME.startswith('roberta'):
            TOKEN_TYPE_IDS1=np.empty((data.shape[0],MAX_LENGTH),dtype=np.int32)
            TOKEN_TYPE_IDS2=np.empty((data.shape[0],MAX_LENGTH),dtype=np.int32)
        for i in range(data.shape[0]):
            t1,t2=h1
            s1=' '.join(data.loc[i,t1].values)
            s2=' '.join(data.loc[i,t2].values)
            if PRE_NAME.startswith('bert'):
                INPUT_IDS1[i,],ATTENTION_MASK1[i,],TOKEN_TYPE_IDS1[i,]=get_bert_single_head_inputs(Tokenizer(PRE_NAME,tokenizer_path),s1,s2,MAX_LENGTH)
            elif PRE_NAME.startswith('xlnet'):
                INPUT_IDS1[i,],ATTENTION_MASK1[i,],TOKEN_TYPE_IDS1[i,]=get_xlnet_single_head_inputs(Tokenizer(PRE_NAME,tokenizer_path),s1,s2,MAX_LENGTH)
            else:
                INPUT_IDS1[i,],ATTENTION_MASK1[i,]=get_roberta_single_head_inputs(Tokenizer(PRE_NAME,tokenizer_path),' '+s1,' '+s2,MAX_LENGTH)
            t1,t2=h2
            s1=' '.join(data.loc[i,t1].values)
            s2=' '.join(data.loc[i,t2].values)
            if PRE_NAME.startswith('bert'):
                INPUT_IDS2[i,],ATTENTION_MASK2[i,],TOKEN_TYPE_IDS2[i,]=get_bert_single_head_inputs(Tokenizer(PRE_NAME,tokenizer_path),s1,s2,MAX_LENGTH)
            elif PRE_NAME.startswith('xlnet'):
                INPUT_IDS2[i,],ATTENTION_MASK2[i,],TOKEN_TYPE_IDS2[i,]=get_xlnet_single_head_inputs(Tokenizer(PRE_NAME,tokenizer_path),s1,s2,MAX_LENGTH)
            else:
                INPUT_IDS2[i,],ATTENTION_MASK2[i,]=get_roberta_single_head_inputs(Tokenizer(PRE_NAME,tokenizer_path),' '+s1,' '+s2,MAX_LENGTH)
            if not inference:
                Y[i,]=data.loc[i,y_columns]
        if not PRE_NAME.startswith('roberta') and not inference:
            return INPUT_IDS1,ATTENTION_MASK1,TOKEN_TYPE_IDS1,INPUT_IDS2,ATTENTION_MASK2,TOKEN_TYPE_IDS2,Y
        if not PRE_NAME.startswith('roberta') and inference:
            return INPUT_IDS1,ATTENTION_MASK1,TOKEN_TYPE_IDS1,INPUT_IDS2,ATTENTION_MASK2,TOKEN_TYPE_IDS2
        else:
            if not inference:
                return INPUT_IDS1,ATTENTION_MASK1,INPUT_IDS2,ATTENTION_MASK2,Y
            else:
                return INPUT_IDS1,ATTENTION_MASK1,INPUT_IDS2,ATTENTION_MASK2


def get_final_model_inputs(HEADS,PRE_NAME,MAX_LENGTH,tokenizer_path,data,h1,h2,inference=False):
    if HEADS==1:
        if not PRE_NAME.startswith('roberta'):
            if not inference:
                INPUT_IDS,ATTENTION_MASK,TOKEN_TYPE_IDS,Y=get_inputs(PRE_NAME,MAX_LENGTH,tokenizer_path,data,h1,h2,inference)
            else:
                INPUT_IDS,ATTENTION_MASK,TOKEN_TYPE_IDS=get_inputs(PRE_NAME,MAX_LENGTH,tokenizer_path,data,h1,h2,inference)
            model_inputs={'input_ids':INPUT_IDS,
                         'attention_mask':ATTENTION_MASK,
                         'token_type_ids':TOKEN_TYPE_IDS}
            if not inference:
                model_outputs=Y
        else:
            if not inference:
                INPUT_IDS,ATTENTION_MASK,Y=get_inputs(PRE_NAME,MAX_LENGTH,tokenizer_path,data,h1,h2,inference)
            else:
                INPUT_IDS,ATTENTION_MASK=get_inputs(PRE_NAME,MAX_LENGTH,tokenizer_path,data,h1,h2,inference)
            model_inputs={'input_ids':INPUT_IDS,
                         'attention_mask':ATTENTION_MASK
                         }
            if not inference:
                model_outputs=Y
    else:
        if not PRE_NAME.startswith('roberta'):
            if not inference:
                INPUT_IDS1,ATTENTION_MASK1,TOKEN_TYPE_IDS1,INPUT_IDS2,ATTENTION_MASK2,TOKEN_TYPE_IDS2,Y=get_inputs(PRE_NAME,MAX_LENGTH,tokenizer_path,data,h1,h2,inference)
            else:
                INPUT_IDS1,ATTENTION_MASK1,TOKEN_TYPE_IDS1,INPUT_IDS2,ATTENTION_MASK2,TOKEN_TYPE_IDS2=get_inputs(PRE_NAME,MAX_LENGTH,tokenizer_path,data,h1,h2,inference)
            model_inputs=[{'input_ids1':INPUT_IDS1,
                         'attention_mask1':ATTENTION_MASK1,
                         'token_type_ids1':TOKEN_TYPE_IDS1},
                          {'input_ids2':INPUT_IDS2,
                         'attention_mask2':ATTENTION_MASK2,
                         'token_type_ids2':TOKEN_TYPE_IDS2}]
            if not inference:
                model_outputs=Y
        else:
            if not inference:
                INPUT_IDS1,ATTENTION_MASK1,INPUT_IDS2,ATTENTION_MASK2,Y=get_inputs(PRE_NAME,MAX_LENGTH,tokenizer_path,data,h1,h2,inference)
            else:
                INPUT_IDS1,ATTENTION_MASK1,INPUT_IDS2,ATTENTION_MASK2=get_inputs(PRE_NAME,MAX_LENGTH,tokenizer_path,data,h1,h2,inference)
            model_inputs=[{'input_ids1':INPUT_IDS1,
                         'attention_mask1':ATTENTION_MASK1},
                          {'input_ids2':INPUT_IDS2,
                         'attention_mask2':ATTENTION_MASK2}]
            if not inference:
                model_outputs=Y
    if not inference:
        return model_inputs,model_outputs
    else:
        return model_inputs