#imports
import os
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
import numpy as np
import random
from nltk.tokenize import word_tokenize
import string

#A class to extract sentences from a text corpus from files inside a folder/directory
class My_Sentence(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

#This function loops through each word in the data set, determines if it is in the vocabulary
#and in that case, adds the mathing integer index to a list.
def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data

class Tokenize_Into_Words(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                    allWords = []
                    tokenized_line = ' '.join(word_tokenize(line))
                    single_sentence = [word for word in tokenized_line.split()]
                    for word in single_sentence:
                        allWords.append(word_tokenize(word.lower()))
                        for wrd in allWords:
                            yield wrd

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"whats", "", text)
    text = re.sub( r"whos", "", text )
    text = re.sub( r"wheres", "", text )
    text = re.sub( r"whens", "", text )
    text = re.sub( r"hows", "", text )

    text = re.sub( r"cant", "", text )
    text = re.sub( r"doesnt", "", text )
    text = re.sub( r"dont", "", text )
    text = re.sub( r"wont", "", text )

    text = re.sub( r"im", "", text )
    text = re.sub( r"youre", "", text )
    text = re.sub( r"were", "", text )
    text = re.sub( r"theyre", "", text )

    text = re.sub( r"id", "", text )
    text = re.sub( r"youd", "", text )
    text = re.sub( r"shed", "", text )
    text = re.sub( r"hed", "", text )
    text = re.sub( r"wed", "", text )
    text = re.sub( r"theyd", "", text )

    text = re.sub( r"ill", "", text )
    text = re.sub( r"youll", "", text )
    text = re.sub( r"shell", "", text )
    text = re.sub( r"he'll", "will", text )
    text = re.sub( r"well", "", text )
    text = re.sub( r"theyll", "", text )

    text = re.sub( r"ive", "", text )
    text = re.sub( r"youve", "", text )
    text = re.sub( r"shes", "", text )
    text = re.sub( r"hes", "", text )
    text = re.sub( r"weve", "", text )
    text = re.sub( r"theyve", "", text )

    text = re.sub( r"havent", "", text )
    text = re.sub( r"hasnt", "", text )
    text = re.sub( r"wouldnt", "", text )
    text = re.sub( r"shouldnt", "", text )

    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)

def create_dictionary():
    """
    create char2id, id2char and vocab_size
    from printable ascii characters.
    """
    chars = sorted(ch for ch in string.printable if ch not in ("\x0b", "\x0c", "\r"))
    char2id = dict((ch, i + 1) for i, ch in enumerate(chars))
    char2id.update({"": 0})
    id2char = dict((char2id[ch], ch) for ch in char2id)
    vocab_size = len(char2id)
    return char2id, id2char, vocab_size

CHAR2ID, ID2CHAR, VOCAB_SIZE = create_dictionary()

def encode_text(text, char2id=CHAR2ID):
    """
    encode text to array of integers with CHAR2ID
    """
    return np.fromiter((char2id.get(ch, 0) for ch in text), int)

def sample_from_probs(probs, top_n=10):
    """
    truncated weighted random choice.
    """
    # need 64 floating point precision
    probs = np.array(probs, dtype=np.float64)
    # set probabilities after top_n to 0
    probs[np.argsort(probs)[:-top_n]] = 0
    # renormalise probabilities
    probs /= np.sum(probs)
    sampled_index = np.random.choice(len(probs), p=probs)
    return sampled_index

def generate_text(model, seed, length=100, top_n=10):
    """
    generates text of specified length from trained model
    with given seed character sequence.
    """
    print('\n'"generating %s characters from top %s choices." % (length, top_n))
    print('generating with seed: "%s".' % seed)
    generated = seed
    encoded = encode_text(seed)
    model.reset_states()

    for idx in encoded[:-1]:
        x = np.array([[idx]])
        # input shape: (1, 1)
        # set internal states
        model.predict(x)
    next_index = encoded[-1]

    for i in range(length):
        x = np.array([[next_index]])
        # input shape: (1, 1)
        probs = model.predict(x)
        # output shape: (1, 1, vocab_size)
        next_index = sample_from_probs(probs.squeeze(), top_n)
        # append to sequence
        generated += ID2CHAR[next_index]

    print("generated text: \n%s\n", generated)
    return generated

def generate_seed(text, seq_lens=(2, 4, 8, 16, 32)):
    # randomly choose sequence length
    seq_len = random.choice(seq_lens)
    # randomly choose start index
    start_index = random.randint(0, len(text) - seq_len - 1)
    seed = text[start_index: start_index + seq_len]
    return seed

def get_sim(valid_word_idx, vocab_size, k_model):
    sim = np.zeros((vocab_size,))
    in_arr1 = np.zeros((1,))
    in_arr2 = np.zeros((1,))
    in_arr1[0,] = valid_word_idx

    for i in range(vocab_size):
        in_arr2[0,] = i
        out = k_model.predict_on_batch([in_arr1, in_arr2])
        sim[i] = out
    return sim

def GetridOfDigits(sentence):
    word = re.sub(r'\d', "", sentence)
    return word

def FileGetRidOfDigits(recipient_file, file_to_clean):
    with open(recipient_file, "w") as text_file:
        for verb in file_to_clean:
            text_file.write("%s" % GetridOfDigits(verb))

