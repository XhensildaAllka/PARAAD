# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential
from keras_self_attention import SeqSelfAttention

class ParaadSa:
    def __init__(self, n_epochs, batch_size, input_shape, learning_rate, drop_out_rate):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.drop_out_rate = drop_out_rate
        self.autoencoder = self._build_PARAAD_Sa_autoencoder()

    def _build_PARAAD_Sa_autoencoder(self):
        # Define the autorncoder
        autoencoder = Sequential()
        S=self.input_shape[1]
        D=self.input_shape[2]
        autoencoder.add(Dropout(self.drop_out_rate, seed = 2))
        autoencoder.add(SeqSelfAttention(attention_width=24, attention_activation='sigmoid'))
        autoencoder.add(Bidirectional(LSTM(7, activation='linear', input_shape=(D,S), return_sequences=True)))
        autoencoder.add(Bidirectional(LSTM(2, activation='relu', return_sequences=False)))
        autoencoder.add(RepeatVector(S))
        autoencoder.add(SeqSelfAttention(attention_width=4, attention_activation='sigmoid'))
        autoencoder.add(Bidirectional(LSTM(7, activation='linear', return_sequences=True)))
        autoencoder.add(TimeDistributed(Dense(D)))
        autoencoder.compile(optimizer = tf.keras.optimizers.Adam(1e-2, clipnorm = 1), loss='mse')
        autoencoder.build(self.input_shape)
        return autoencoder    


    def train(self, train_data):
        self.autoencoder.fit(train_data, train_data,
                             epochs=self.n_epochs,
                             batch_size=self.batch_size,
                             shuffle=False)

    def predict(self, test_data, verbose=0):
        reconstructed_data = self.autoencoder.predict(test_data)
        return reconstructed_data
