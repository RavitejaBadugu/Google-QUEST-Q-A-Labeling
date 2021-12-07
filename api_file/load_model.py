import tensorflow as tf


class MODELS:
    def __init__(self,model_fn,paths):
        print(paths[0])
        self.model_1=model_fn.load_weights(paths[0])
        self.model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
                 loss='binary_crossentropy')
        print(f"model1 {self.model_1.summary()}")
        self.model2=model_fn.load_weights(paths[1])
        print(f"model2 {self.model2.summary()}")
        self.model3=model_fn.load_weights(paths[2])
        self.model4=model_fn.load_weights(paths[3])
        self.model5=model_fn.load_weights(paths[4])
        
    def predict(self,model_inputs):
        print(f"model1 {self.model1.summary()}")
        p1=self.model1.predict(model_inputs)
        print(f"model2 {self.model2.summary()}")
        p2=self.model2.predict(model_inputs)
        p3=self.model3.predict(model_inputs)
        p4=self.model4.predict(model_inputs)
        p5=self.model5.predict(model_inputs)
        p=(p1+p2+p3+p4+p5)/5
        return p