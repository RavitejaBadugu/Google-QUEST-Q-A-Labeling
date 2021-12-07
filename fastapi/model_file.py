from tensorflow.keras.layers import Dense,Dropout,Input,Conv1D,BatchNormalization
from transformers import TFBertModel,TFRobertaModel,TFXLNetModel
import tensorflow as tf

def SINGLE_MODEL(PRE_NAME,MAX_LENGTH,sequence=False,final_activation=True,hidden_states=True,hidden_number=4):
    tf.keras.backend.clear_session()
    if PRE_NAME.startswith('bert'):
        ins1=Input((MAX_LENGTH,),dtype=tf.int32)
        ins2=Input((MAX_LENGTH,),dtype=tf.int32)
        ins3=Input((MAX_LENGTH,),dtype=tf.int32)
        pre_model=TFBertModel.from_pretrained(PRE_NAME,output_hidden_states=hidden_states,return_dict=True)
        pre_layers=pre_model({'input_ids':ins1,'attention_mask':ins2,'token_type_ids':ins3})
    elif PRE_NAME.startswith('xlnet'):
        ins1=Input((MAX_LENGTH,),dtype=tf.int32)
        ins2=Input((MAX_LENGTH,),dtype=tf.int32)
        ins3=Input((MAX_LENGTH,),dtype=tf.int32)
        pre_model=TFXLNetModel.from_pretrained(PRE_NAME,output_hidden_states=hidden_states,return_dict=True)
        pre_layers=pre_model({'input_ids':ins1,'attention_mask':ins2,'token_type_ids':ins3})
    else:
        ins1=Input((MAX_LENGTH,),dtype=tf.int32)
        ins2=Input((MAX_LENGTH,),dtype=tf.int32)
        pre_model=TFRobertaModel.from_pretrained(PRE_NAME,output_hidden_states=hidden_states,return_dict=True)
        pre_layers=pre_model({'input_ids':ins1,'attention_mask':ins2})
    if sequence:
        x=Conv1D(1,1)(pre_layers[0])
        x=tf.squeeze(x,axis=-1)
        x=BatchNormalization()(x)
        x=Dropout(0.1)(x)
        x=tf.keras.layers.ReLU()(x)
    elif hidden_states:
        if not PRE_NAME.startswith('xlnet'):
            x=tf.stack([layer[:,0,:] for layer in pre_layers[2][-hidden_number:]],axis=-1)
            x=tf.keras.layers.Flatten()(x)
            x=Dense(768*hidden_number,activation='tanh')(x)
        else:
            x=tf.stack([layer[:,-1,:] for layer in pre_layers[2][-hidden_number:]],axis=-1)
            x=tf.keras.layers.Flatten()(x)
            x=Dense(768*hidden_number,activation='tanh')(x)
    else:
        if not PRE_NAME.startswith('xlnet'):
            x=pre_layers[1]
        else:
            x=pre_layers[0][:,-1,:]
        x=BatchNormalization()(x)
        x=Dropout(0.1)(x)
    if final_activation:
        outs=Dense(30,activation='sigmoid')(x)
    else:
        outs=Dense(30)(x)
    if not PRE_NAME.startswith('roberta'):
        model=tf.keras.models.Model(inputs={'input_ids':ins1,'attention_mask':ins2,'token_type_ids':ins3},outputs=outs)
    else:
        model=tf.keras.models.Model(inputs={'input_ids':ins1,'attention_mask':ins2},outputs=outs)
    return model


def DOUBLE_MODEL(PRE_NAME,MAX_LENGTH,sequence=False,final_activation=True,hidden_states=True):
    tf.keras.backend.clear_session()
    if PRE_NAME.startswith('bert'):
        ins1=Input((MAX_LENGTH,),dtype=tf.int32)
        ins2=Input((MAX_LENGTH,),dtype=tf.int32)
        ins3=Input((MAX_LENGTH,),dtype=tf.int32)
        ins4=Input((MAX_LENGTH,),dtype=tf.int32)
        ins5=Input((MAX_LENGTH,),dtype=tf.int32)
        ins6=Input((MAX_LENGTH,),dtype=tf.int32)
        pre_model=TFBertModel.from_pretrained(PRE_NAME,output_hidden_states=hidden_states,return_dict=True)
        pre_layers1=pre_model({'input_ids':ins1,'attention_mask':ins2,'token_type_ids':ins3})
        pre_layers2=pre_model({'input_ids':ins4,'attention_mask':ins5,'token_type_ids':ins6})
    elif PRE_NAME.startswith('xlnet'):
        ins1=Input((MAX_LENGTH,),dtype=tf.int32)
        ins2=Input((MAX_LENGTH,),dtype=tf.int32)
        ins3=Input((MAX_LENGTH,),dtype=tf.int32)
        ins4=Input((MAX_LENGTH,),dtype=tf.int32)
        ins5=Input((MAX_LENGTH,),dtype=tf.int32)
        ins6=Input((MAX_LENGTH,),dtype=tf.int32)
        pre_model=TFXLNetModel.from_pretrained(PRE_NAME,output_hidden_states=hidden_states,return_dict=True)
        pre_layers1=pre_model({'input_ids':ins1,'attention_mask':ins2,'token_type_ids':ins3})
        pre_layers2=pre_model({'input_ids':ins4,'attention_mask':ins5,'token_type_ids':ins6})
    else:
        ins1=Input((MAX_LENGTH,),dtype=tf.int32)
        ins2=Input((MAX_LENGTH,),dtype=tf.int32)
        ins3=Input((MAX_LENGTH,),dtype=tf.int32)
        ins4=Input((MAX_LENGTH,),dtype=tf.int32)
        pre_model=TFRobertaModel.from_pretrained(PRE_NAME,output_hidden_states=hidden_states,return_dict=True)
        pre_layers1=pre_model({'input_ids':ins1,'attention_mask':ins2})
        pre_layers2=pre_model({'input_ids':ins3,'attention_mask':ins4})
    if sequence:
        x1=Conv1D(1,1)(pre_layers1[0])
        x1=tf.squeeze(x1,axis=-1)
        x1=BatchNormalization()(x1)
        x1=Dropout(0.1)(x1)
        x1=tf.keras.layers.ReLU()(x1)
        x2=Conv1D(1,1)(pre_layers2[0])
        x2=tf.squeeze(x2,axis=-1)
        x2=BatchNormalization()(x2)
        x2=Dropout(0.1)(x2)
        x2=tf.keras.layers.ReLU()(x2)
        x=tf.keras.layers.Concatenate()([x1,x2])
    elif hidden_states:
        if not PRE_NAME.startswith('xlnet'):
            x1=tf.stack([layer[:,0,:] for layer in pre_layers1[2]],axis=-1)
            x1=tf.squeeze(Conv1D(1,1)(x1),axis=-1)
            x1=BatchNormalization()(x1)
            x1=Dropout(0.1)(x1)
            x1=tf.keras.layers.ReLU()(x1)
            x2=tf.stack([layer[:,0,:] for layer in pre_layers2[2]],axis=-1)
            x2=tf.squeeze(Conv1D(1,1)(x2),axis=-1)
            x2=BatchNormalization()(x2)
            x2=Dropout(0.1)(x2)
            x2=tf.keras.layers.ReLU()(x2)
        else:
            x1=tf.stack([layer[:,-1,:] for layer in pre_layers1[2]],axis=-1)
            x1=tf.squeeze(Conv1D(1,1)(x1),axis=-1)
            x1=BatchNormalization()(x1)
            x1=Dropout(0.1)(x1)
            x1=tf.keras.layers.ReLU()(x1)
            x2=tf.stack([layer[:,-1,:] for layer in pre_layers2[2]],axis=-1)
            x2=tf.squeeze(Conv1D(1,1)(x2),axis=-1)
            x2=BatchNormalization()(x2)
            x2=Dropout(0.1)(x2)
            x2=tf.keras.layers.ReLU()(x2)
        x=tf.keras.layers.Concatenate()([x1,x2])
    else:
        if not PRE_NAME.startswith('xlnet'):
            x1=pre_layers1[1]
            x1=BatchNormalization()(x1)
            x1=Dropout(0.1)(x1)
            x2=pre_layers2[1]
            x2=BatchNormalization()(x2)
            x2=Dropout(0.1)(x2)
        else:
            x1=pre_layers1[0][:,-1,:]
            x1=BatchNormalization()(x1)
            x1=Dropout(0.1)(x1)
            x2=pre_layers2[0][:,-1,:]
            x2=BatchNormalization()(x2)
            x2=Dropout(0.1)(x2)
        x=tf.keras.layers.Concatenate()([x1,x2])
    if final_activation:
        outs=Dense(30,activation='sigmoid')(x)
    else:
        outs=Dense(30)(x)
    if not PRE_NAME.startswith('roberta'):
        model=tf.keras.models.Model(inputs=[{'input_ids1':ins1,'attention_mask1':ins2,'token_type_ids1':ins3},
                                            {'input_ids2':ins4,'attention_mask2':ins5,'token_type_ids2':ins6}]
                                    ,outputs=outs)
    else:
        model=tf.keras.models.Model(inputs=[{'input_ids1':ins1,'attention_mask1':ins2},
                                            {'input_ids2':ins3,'attention_mask2':ins4}]
                                    ,outputs=outs)
    
    return model