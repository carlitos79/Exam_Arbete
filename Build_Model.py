from keras.models import load_model, Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from keras.optimizers import Adam
from Utils import *

def build_model(batch_size, seq_len, vocab_size=VOCAB_SIZE,
                embedding_size=32,
                rnn_size=128,
                num_layers=2,
                drop_rate=0.0,
                learning_rate=0.001,
                clip_norm=5.0):
    #build character embeddings LSTM text generation model.
    model = Sequential()
    # input shape: (batch_size, seq_len)
    model.add(Embedding(vocab_size, embedding_size, batch_input_shape=(batch_size, seq_len)))
    model.add(Dropout(drop_rate))
    # shape: (batch_size, seq_len, embedding_size)
    for _ in range(num_layers):
        model.add(LSTM(rnn_size, return_sequences=True, stateful=True))
        model.add(Dropout(drop_rate))
    # shape: (batch_size, seq_len, rnn_size)
    model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))
    # output shape: (batch_size, seq_len, vocab_size)
    optimizer = Adam(learning_rate, clipnorm=clip_norm)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model