######################################## IMPORTS ###################################################
import warnings;
warnings.filterwarnings(action = 'ignore', category = UserWarning, module = 'gensim')
warnings.filterwarnings(action = 'ignore', category = UserWarning, module = 'keras')
from time import time
from gensim.models.keyedvectors import KeyedVectors
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, TimeDistributed
from keras.preprocessing.text import Tokenizer, one_hot
import numpy as np
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers import merge
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Utils import *

########################################## DIRECTORIES #################################################
base_dir = "C:/Users/Carlos Peñaloza/Desktop/No-Backup Zone/RNN_With_Embeddings/"
corpus = base_dir + "word2vec_model"
nietzche = "C:/Users/Carlos Peñaloza/Desktop/No-Backup Zone/RNN_With_Embeddings/Nietzche/Nietzche.txt"

###################################### PARAMETERS #######################################
vector_dimension = 300
batch_size = 32
rnn_size = 128
learning_rate= 0.001
clip_norm = 5.0
act = 'relu'
number_of_layers = 2
max_sequence_length = 30
max_number_of_words = 100
number_of_lstm = np.random.randint(175, 275)
number_of_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25
STAMP = 'lstm_%d_%d_%.2f_%.2f'%(number_of_lstm, number_of_dense, rate_drop_lstm, rate_drop_dense)

###################################### LOAD MODEL ########################################
print("Loading...")
begin = time()
project_model = KeyedVectors.load_word2vec_format(corpus)
end = time()
seconds = end - begin
minutes = seconds / 60
print("Done loading.")
print("Model loaded after:")
print("Time in seconds: %d" %seconds)
print("Time in minutes: %d" %minutes)
print('\n'"%s word vectors of word2vec found" % len(project_model.vocab))

################################### EMBEDDING LAYER #######################################
vocabulary_size = len(project_model.vocab)

embedding_matrix = np.zeros((vocabulary_size, vector_dimension))
for word in range(vocabulary_size):
    if word in project_model.wv.vocab:
        embedding_matrix[word] = project_model.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

#embedding_matrix = np.zeros((len(project_model.wv.vocab), vector_dimension))
#for i in range(len(project_model.wv.vocab)):
#    embedding_vector = project_model.wv[project_model.wv.index2word[i]]
#    if embedding_vector is not None:
#        embedding_matrix[i] = embedding_vector

################################### MODEL STRUCTURE #######################################
embedding_layer = Embedding(vocabulary_size,
                            vector_dimension,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=False)

lstm_layer = LSTM(number_of_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(max_sequence_length,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(max_sequence_length,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(number_of_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

#################################### MODEL TRAINING ##########################################

model = Model(input=[sequence_1_input, sequence_2_input], output=preds)
model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['acc'])
model.summary()
print(STAMP)

##############################################################################################


