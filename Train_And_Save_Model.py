#imports
from time import time
from Utils import *
import warnings; warnings.filterwarnings(action = 'ignore', category = UserWarning, module = 'gensim')
import gensim
import logging
import multiprocessing
from keras.utils.data_utils import get_file
from nltk.tokenize import sent_tokenize, word_tokenize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

path_to_data = "C:/Users/Carlos Peñaloza/Desktop/No-Backup Zone/RNN_With_Embeddings/Nietzche"


print("Training...")
begin = time()
sentences = My_Sentence(path_to_data)
project_model = gensim.models.Word2Vec(sentences, min_count=10, size=300, workers=multiprocessing.cpu_count())
end = time()
seconds = end - begin
minutes = seconds / 60
print("Done training.")
print("Total processing time in seconds: %d" %seconds)
print("Total processing time in minutes: %d" %minutes)

base_dir = "C:/Users/Carlos Peñaloza/Desktop/No-Backup Zone/RNN_With_Embeddings/"

regular_model = base_dir + "regular_model"
word2vec_model = base_dir + "word2vec_model" #we use this one
vocabulary = base_dir + "vocabulary"

project_model.save(regular_model)
project_model.wv.save_word2vec_format(word2vec_model, vocabulary)