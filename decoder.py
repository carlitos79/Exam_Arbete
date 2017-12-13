import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils import plot_model

# Decoder

class Decoder():
    def __init__(self, encoder, n_tokens, max_seq_length):
        self.encoder = encoder
        self.n_tokens = n_tokens
        self.max_seq_length = max_seq_length
        self.token_to_index = dict()
        self.index_to_token = dict()

        self.decoder_inputs = Input(shape=(None, self.n_tokens))
        self.decoder_lstm = LSTM(self.encoder.dimension,
                                 return_sequences=True,
                                 return_state=True)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs)
        self.decoder_dense = Dense(self.n_tokens, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        self.decoder_model = Model(self.decoder_inputs, self.decoder_outputs)

    def fit(self, X, y, *args, **kwargs):
        target_chars = sorted(list(set(''.join(y))))
        self.token_to_index = dict([(c, i) for i, c in enumerate(target_chars)])
        self.index_to_token = dict([(i, c) for i, c, in enumerate(target_chars)])

        decoder_input_data = np.zeros(
            (self.encoder.dimension, self.max_seq_length, self.n_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (self.encoder.dimension, self.max_seq_length, self.n_tokens),
            dtype='float32')
        for i, target in enumerate(y):
            for t, char in enumerate(target):
                decoder_input_data[i, t, self.token_to_index[char]] = 1
                if t > 0:
                    decoder_target_data[i, t-1, self.token_to_index[char]] = 1

        self.decoder_model.compile(optimizer='rmsprop',
                                   loss='categorical_crossentropy')
        seeds = np.array([np.reshape(self.encoder.encode(s),
                                     (1, self.encoder.dimension))
                          for s in X])
        hidden_states = K.variable(value=seeds)
        cell_states = K.variable(value=seeds)
        self.decoder_model.layers[1].states[0] = hidden_states
        self.decoder_model.layers[1].states[1] = cell_states
        self.decoder_model.fit(decoder_input_data,
                               decoder_target_data,
                               *args,
                               **kwargs)

    def decode(self, X):
        decoder_state_input_h = Input(shape=(self.encoder.dimension,))
        decoder_state_input_c = Input(shape=(self.encoder.dimension,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        inference_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        res = []
        for x in X:
            state_h = np.reshape(self.encoder.encode(x),
                                 (1, self.encoder.dimension))
            state_c = np.reshape(self.encoder.encode(x),
                                 (1, self.encoder.dimension))

            target = np.zeros((1, 1, self.n_tokens))
            target[0, 0, self.token_to_index['\t']] = 1
            stop_cond = False
            decoded_sequence = ''
            while not stop_cond:
                output, h, c = inference_model.predict([target]
                                                     + [state_h, state_c])
                pred_index = np.argmax(output[0, -1, :])
                pred_char = self.index_to_token[pred_index]
                decoded_sequence += pred_char

                if(pred_char == '\n' or
                           len(decoded_sequence) > self.max_seq_length):
                    stop_cond = True

                target = np.zeros((1, 1, self.n_tokens))
                target[0, 0, pred_index] = 1
                state_h = h
                state_c = c
            res.append(decoded_sequence)
        return np.array(res)

    def save(self, f_name, *args, **kwargs):
        self.decoder_model.save(f_name, *args, **kwargs)

    def load(self, f_name, *args, **kwargs):
        self.decoder_model.load_weights(f_name, *args, **kwargs)

    def plot(self, f_name):
        plot_model(self.decoder_model, f_name)

    def summary(self):
        return self.decoder_model.summary()