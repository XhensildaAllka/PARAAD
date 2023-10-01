# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.models import Model

class ParaadMHa:
    def __init__(self, n_heads, n_epochs, batch_size, input_shape, learning_rate, drop_out_rate):
        self.n_heads = n_heads
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.drop_out_rate = drop_out_rate
        self.autoencoder = self._build_PARAAD_MHa_autoencoder()

    def _build_PARAAD_MHa_autoencoder(self):
        # Define the autorncoder
        
        S=self.input_shape[0]
        D=self.input_shape[1]
        inputs = Input(shape=self.input_shape)
        x = Dropout(self.drop_out_rate, seed=3)(inputs)
        x = MultiHeadAttention(num_heads = self.n_heads, key_dim=24)(x, x)
        x = Bidirectional(LSTM(7, activation='linear', input_shape=(D, S), return_sequences=True))(x)
        x = Bidirectional(LSTM(2, activation='relu', return_sequences=False))(x)
        x = RepeatVector(S)(x)
        x = MultiHeadAttention(num_heads=self.n_heads, key_dim=4)(x, x)
        x = Bidirectional(LSTM(7, activation='linear', return_sequences=True))(x)
        outputs = TimeDistributed(Dense(D))(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer='adam', loss='mse')
        return model    


    def train(self, train_data):
        self.autoencoder.fit(train_data, train_data,
                             epochs=self.n_epochs,
                             batch_size=self.batch_size,
                             shuffle=False)

    def predict(self, test_data, verbose=0):
        reconstructed_data = self.autoencoder.predict(test_data)
        return reconstructed_data
