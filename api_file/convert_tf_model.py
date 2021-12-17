import tensorflow as tf
from api_file.model_file import SINGLE_MODEL
from api_file.utils.config_reader import inference_parameters
import gc

class Save_Model:
    def save_models(self,paths,to_paths):
        for i in range(len(paths)):
            tf.keras.backend.clear_session()
            model=SINGLE_MODEL(PRE_NAME=inference_parameters['PRE_NAME'],
                                  MAX_LENGTH=inference_parameters['MAX_LENGTH'],
                                  PRE_MODEL=inference_parameters['PRE_MODEL'],
                                  sequence=False,
                                  final_activation=True,
                                  hidden_states=True, 
                                  hidden_number=4)
            model.load_weights(paths[i])
            tf.saved_model.save(model,to_paths[i])
            del model
            gc.collect()
            

if __name__ == '__main__':
    saving=Save_Model()
    saving.save_models(['tf-serving/results/bert_4_hiddens_type2_bce_fold_0.h5',
                        'tf-serving/results/bert_4_hiddens_type2_bce_fold_1.h5',
                        'tf-serving/results/bert_4_hiddens_type2_bce_fold_2.h5',
                        'tf-serving/results/bert_4_hiddens_type2_bce_fold_3.h5',
                        'tf-serving/results/bert_4_hiddens_type2_bce_fold_4.h5'],
                        ['tf-serving/results/serving_models/fold0/v1/',
                        'tf-serving/results/serving_models/fold1/v1/',
                        'tf-serving/results/serving_models/fold2/v1/',
                        'tf-serving/results/serving_models/fold3/v1/',
                        'tf-serving/results/serving_models/fold4/v1/'])
