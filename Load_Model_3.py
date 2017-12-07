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
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers import merge
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Utils import *
import sys

########################################## DIRECTORIES #################################################
base_dir = "C:/Users/Carlos Peñaloza/Desktop/No-Backup Zone/RNN_With_Embeddings/"
corpus = base_dir + "word2vec_model"
nietzche_path = "C:/Users/Carlos Peñaloza/Desktop/No-Backup Zone/RNN_With_Embeddings/Nietzche/Nietzche.txt"
text = open(nietzche_path).read().lower()

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

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=embedding_matrix.shape[0],
                    output_dim=embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    input_length=max_sequence_length,
                    trainable=False))
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):

    print('Iteration', iteration)
    model.fit(x, y, batch_size=128, epochs=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += ""
        print('----- Generating with seed: "' + sentence + '"')
        print("RESULT: ")
        sys.stdout.write(generated)

#############################################################################

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars))) #### code to change
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
print()