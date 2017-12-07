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

embedding_matrix = np.zeros((len(project_model.wv.vocab), vector_dimension))
for i in range(len(project_model.wv.vocab)):
    embedding_vector = project_model.wv[project_model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

valid_size = 5  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# input words - in this case we do sample by sample evaluations of the similarity
valid_word = Input((1,), dtype='int32')
other_word = Input((1,), dtype='int32')

# setup the embedding layer
embeddings = Embedding(input_dim=embedding_matrix.shape[0],
                       output_dim=embedding_matrix.shape[1],
                       weights=[embedding_matrix])

embedded_a = embeddings(valid_word)
embedded_b = embeddings(other_word)
similarity = merge([embedded_a, embedded_b], mode='cos', dot_axes=2)

############################# KERAS MODEL ################################
k_model = Model(input=[valid_word, other_word], output=similarity)
#k_model.summary()

def get_sim(valid_word_idx, vocab_size):
    sim = np.zeros((vocab_size,))
    in_arr1 = np.zeros((1,))
    in_arr2 = np.zeros((1,))
    in_arr1[0,] = valid_word_idx

    for i in range(vocab_size):
        in_arr2[0,] = i
        out2 = k_model.predict([in_arr1, in_arr2], vocabulary_size, verbose=0)[0]
        out = k_model.predict_on_batch([in_arr1, in_arr2])
        sim[i] = out
    return sim

#######################################################################
for i in range(valid_size):
    valid_word = project_model.wv.index2word[valid_examples[i]]
    top_k = 5  # number of nearest neighbors
    sim = get_sim(valid_examples[i], len(project_model.wv.vocab))
    nearest = (-sim).argsort()[1:top_k + 1]
    log_str = 'Nearest to %s:' % valid_word

    for k in range(top_k):
        close_word = project_model.wv.index2word[nearest[k]]
        log_str = '%s %s,' % (log_str, close_word)

    print(log_str)

#######################################################################