import math
import encoder
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Embedding
from keras.layers.merge import add
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from warnings import showwarning
from pickle import dump, load


class Decoder():
    '''
    Class representing the concept of a decoder capable of performing
    sequence predictions (decoding) given seeds and an encoder object.
    '''
    def __init__(self, encoder):
        '''
        Initializes the decoder object.
        :param encoder: An encoder object capable of transforming
                        seeds into vector representations.
        '''
        self.encoder = encoder
        self.tokenizer = Tokenizer()
        self.model = None
        self.max_length = 0
        self.vocab_size = 0

    def fit_generator(self, X, y, epochs=25, batch_size=32, trace=True):
        '''
        Fits the model from a generator.
        :param X: An iterable of seeds, each represented as a string.
        :param y: An iterable of targets, each represented as a string.
        :param epochs: Number of training epochs.
        :param batch_size: Size of each training batch.
        :param trace: Flag indicating whether to trace progress.
        '''
        targets = ['startseq ' + t + ' endseq' for t in np.atleast_1d(y)]
        self.tokenizer.fit_on_texts(targets)
        self.max_length = max(len(t.split()) for t in targets)
        self.vocab_size = len(self.tokenizer.word_index) + 1

        if self.model is None:
            self._init_model(self.vocab_size)
        steps = math.ceil((len(y) * self.max_length) / batch_size)
        self.model.fit_generator(self._gen(X, targets, batch_size, steps),
                                 epochs=epochs,
                                 verbose=trace,
                                 steps_per_epoch=steps)

    def fit(self, X, y, epochs=25, batch_size=32, trace=True):
        '''
        Fits the model.
        :param X: An iterable of seeds.
        :param y: An iterable of target sequences.
        :param epochs: Number of training epochs.
        :param batch_size: Size of each training batch.
        :param trace: Flag indicating whether to trace progress.
        '''
        targets = ['startseq ' + t + ' endseq' for t in np.atleast_1d(y)]
        self.tokenizer.fit_on_texts(targets)
        self.max_length = max(len(t.split()) for t in targets)
        self.vocab_size = len(self.tokenizer.word_index) + 1

        X1, X2, y = list(), list(), list()
        for seed, target in zip(np.atleast_1d(X), np.atleast_1d(targets)):
            seq = self.tokenizer.texts_to_sequences([target])[0]
            for i in range(1, len(seq)):
                in_seq = pad_sequences([seq[:i]], maxlen=self.max_length)[0]
                out_seq = to_categorical([seq[i]],
                                         num_classes=self.vocab_size)[0]
                X1.append(self.encoder.encode(seed)[0])
                X2.append(in_seq)
                y.append(out_seq)
        X1 = np.array(X1)
        X2 = np.array(X2)
        y = np.array(y)

        if self.model is None:
            self._init_model(self.vocab_size)

        self.model.fit([X1, X2], y, epochs=epochs,
                       batch_size=batch_size, verbose=trace)

    def decode(self, seeds):
        '''
        Decodes a set of seeds into target sequences.
        :param X: An iterable of seeds.
        :return: An np.ndarray of target sequences.
        '''
        res = []
        sos = 'startseq'
        sos_len = len(sos)
        eos = 'endseq'
        for seed in np.atleast_1d(seeds):
            if self.encoder.can_encode(seed):
                vec = self.encoder.encode(seed)[0].reshape((1, self.encoder.size))
                target = sos
                for i in range(self.max_length):
                    seq = self.tokenizer.texts_to_sequences([target])[0]
                    seq = pad_sequences([seq], maxlen=self.max_length)
                    word = self._word(np.argmax(self.model.predict([vec, seq])))
                    if word is None or word == eos:
                        break
                    target += ' ' + word
                res.append(target[sos_len:])
            else:
                showwarning("Unable to encode seed '{}'".format(seed),
                            category=RuntimeWarning,
                            filename='decoder.py',
                            lineno=99)
        return np.array(res)

    def plot(self, f_name):
        '''
        Creates a plot of the model.
        :param f_name: The target path of the plot.
        '''
        plot_model(self.model, f_name)

    def summary(self):
        '''
        :return: A summary of the model.
        '''
        return self.model.summary()

    def save(self, f_names):
        """
        Saves a model to file.
        :param f_names: A tuple of filenames (root_path, weights_path)
                        to save the model to.
        """
        self.model.save_weights(f_names[1])
        temp = self.model
        self.model = None
        dump(self, open(f_names[0], 'wb'))
        self.model = temp

    @staticmethod
    def load(f_names):
        '''
        Loads a model from file.
        :param f_names: A tuple of paths (root_path, weights_path)
                        to the saved model.
        :return: The loaded model.
        '''
        dec = load(open(f_names[0], 'rb'))
        dec._init_model(dec.vocab_size)
        dec.model.load_weights(f_names[1])
        return dec

    def _word(self, id):
        '''
        Retrieves the word at index id.
        :param id: The index of the word.
        :return: The word at index id.
        '''
        for word, index in self.tokenizer.word_index.items():
            if index == id:
                return word
        return None

    def _init_model(self, vocab_size):
        '''
        Initializes the underlying model.
        :param vocab_size: Size of the vocabulary.
        '''
        latent_dim = 64
        seed_input = Input(shape=(self.encoder.size,))
        seed_dense = Dense(latent_dim, activation='relu')(seed_input)
        gen_input = Input(shape=(self.max_length,))
        gen_embed = Embedding(vocab_size, latent_dim, mask_zero=True)(
            gen_input)
        gen_lstm = LSTM(latent_dim)(gen_embed)
        dec_inputs = add([seed_dense, gen_lstm])
        dec_inputs_dropout = Dropout(0.2)(dec_inputs)
        dec_dense = Dense(latent_dim, activation='relu')(dec_inputs_dropout)
        dec_hidden_dropout = Dropout(0.5)(dec_dense)
        dec_outputs = Dense(vocab_size, activation='softmax')(dec_hidden_dropout)
        self.model = Model(inputs=[seed_input, gen_input], outputs=dec_outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def _gen(self, seeds, targets, batch_size, steps):
        '''
        Helper method for batch generation.
        :param seeds: An iterable of seeds, each represented as a string.
        :param targets: An iterable of targets, each represented as a string.
        :param batch_size: The size of the batch to be generated.
        :return: A tuple ([X1, X2], y) where [X1, X2] is the input
                 and y is the target of the batch.
        '''
        while True:
            for i in range(steps):
                X1, X2, y = list(), list(), list()
                ignored = {}
                for seed, target in zip(seeds, targets):
                    seq = self.tokenizer.texts_to_sequences([target])[0]
                    if self.encoder.can_encode(seed):
                        vec = self.encoder.encode(seed)[0]
                        for i in range(1, len(seq)):
                            in_seq = pad_sequences([seq[:i]], maxlen=self.max_length)[0]
                            out_seq = to_categorical([seq[i]], num_classes=self.vocab_size)[0]
                            X1.append(vec)
                            X2.append(in_seq)
                            y.append(out_seq)
                            if len(X1) == batch_size:
                                for k in ignored:
                                    showwarning("Unable to encode seed '{}', "
                                                "ignored {} instances"
                                                .format(k, ignored[k]),
                                                category=RuntimeWarning,
                                                filename='decoder.py',
                                                lineno=202)
                                yield [np.array(X1), np.array(X2)], np.array(y)
                                X1, X2, y = list(), list(), list()
                    else:
                        if seed in ignored:
                            ignored[seed] += 1
                        else:
                            ignored[seed] = 1


class Decoder_old():
    '''
    Class representing the concept of a decoder capable of performing
    sequence predictions (decoding) given seeds and an encoder object.
    '''
    def __init__(self, encoder, n_tokens, max_seq_length):
        '''
        Initializes the decoder object.
        :param encoder: An encoder object capable of transforming
                        seeds into vector representations.
        :param n_tokens: Number of unique tokens in the target sequences.
        :param max_seq_length: Length of the longest target sequence.
        '''
        self.encoder = encoder
        self.n_tokens = n_tokens
        self.max_seq_length = max_seq_length
        self.token_to_index = dict()
        self.index_to_token = dict()

        self.decoder_inputs = Input(shape=(None, self.n_tokens))
        self.decoder_lstm = LSTM(self.encoder.size,
                                 return_sequences=True,
                                 return_state=True)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs)
        self.decoder_dense = Dense(self.n_tokens, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        self.decoder_model = Model(self.decoder_inputs, self.decoder_outputs)

    def fit(self, X, y, *args, **kwargs):
        '''
        Fits the model.
        :param X: An iterable of seeds.
        :param y: An iterable of target sequences.
        :param args: Forwarding parameters.
        :param kwargs: Forwarding parameters.
        '''
        target_chars = sorted(list(set(''.join(y))))
        self.token_to_index = dict([(c, i) for i, c in enumerate(target_chars)])
        self.index_to_token = dict([(i, c) for i, c, in enumerate(target_chars)])

        decoder_input_data = np.zeros(
            (len(y), self.max_seq_length, self.n_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(y), self.max_seq_length, self.n_tokens),
            dtype='float32')
        for i, target in enumerate(y):
            for t, char in enumerate(target):
                decoder_input_data[i, t, self.token_to_index[char]] = 1
                if t > 0:
                    decoder_target_data[i, t-1, self.token_to_index[char]] = 1

        self.decoder_model.compile(optimizer='rmsprop',
                                   loss='categorical_crossentropy')

        seeds = np.array([np.reshape(self.encoder.encode(s),
                                     (1, self.encoder.size))
                          for s in X])
        hidden_states = K.variable(value=seeds)
        cell_states = K.variable(value=seeds)
        self.decoder_model.layers[1].states[0] = hidden_states
        #self.decoder_model.layers[1].states[1] = cell_states
        self.decoder_model.fit(decoder_input_data,
                               decoder_target_data,
                               *args,
                               **kwargs)

    def decode(self, X):
        '''
        Decodes an iterable of seeds into target sequences.
        :param X: An iterable of seeds.
        :return: An iterable of sequences.
        '''
        decoder_state_input_h = Input(shape=(self.encoder.size,))
        decoder_state_input_c = Input(shape=(self.encoder.size,))
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
                                 (1, self.encoder.size))
            state_c = np.zeros((1, self.encoder.size))
            #state_c = np.reshape(self.encoder.encode(x),
            #                     (1, self.encoder.size))

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
        '''
        Saves the parameters of the trained model.
        :param f_name: The target path.
        :param args: Forwarding parameters.
        :param kwargs: Forwarding parameters.
        '''
        self.decoder_model.save(f_name, *args, **kwargs)

    def load(self, f_name, *args, **kwargs):
        '''
        Loads the parameters of a pre-trained model.
        :param f_name: The source path.
        :param args: Forwarding parameters.
        :param kwargs: Forwarding parameters.
        '''
        self.decoder_model.load_weights(f_name, *args, **kwargs)

    def plot(self, f_name):
        '''
        Creates a plot of the model.
        :param f_name: The target path of the plot.
        '''
        plot_model(self.decoder_model, f_name)

    def summary(self):
        '''
        :return: A summary of the model.
        '''
        return self.decoder_model.summary()