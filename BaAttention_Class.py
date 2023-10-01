# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Layer
from keras import backend as K


class BaAttention(Layer):
   def __init__(self, return_sequences=True):
       self.return_sequences = return_sequences
       super(BaAttention,self).__init__()

   def build(self, input_shape):
       # initializer can be: 'normal'/"glorot_uniform"    
        self.W=self.add_weight(name="BaAtt_weight", shape=(input_shape[-1],1), initializer="glorot_uniform")
        self.b=self.add_weight(name="BaAtt_bias", shape=(input_shape[1],1),
                               initializer="glorot_uniform")
        super(BaAttention,self).build(input_shape) 
   
   def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
           return output
        return K.sum(output, axis=1)
    
   def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences 
        })
        return config