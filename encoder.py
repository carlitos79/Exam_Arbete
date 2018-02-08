import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from itertools import chain
import pickle

'''
Encoders:

Encoder                     - Base class for all concrete encoders
OneHotEncoder               - Encodes a word into a sparse representation
Word2VecEncoder             - Encodes a word into a dense representation
MultiHotEncoder             - Encodes a sentence into a sparse representation
MultiWord2VecEncoder        - Encodes a sentence into a dense representation
'''


class Encoder():
    '''
    Abstract class representing the concept of an encoder.
    '''
    def __repr__(self):
        return self.__class__.__name__

    def encode(self, X):
        '''
        Encodes a token into a vector representation.
        :param X: An iterable of appropriate tokens.
        :return: A vector representation of X.
        '''
        raise NotImplementedError('abstract function {}.{}'
                                  '(X) has not been implemented'
                                  .format(self.__class__.__name__,
                                          self.encode.__name__))

    def can_encode(self, seed):
        '''
        Returns whether or not a given seed can be encoded properly.
        :param seed: The seed for which to test the encodability.
        :return: True if the given seed can be encoded.
        '''
        raise NotImplementedError('abstract function {}.{} '
                                  'has not been implemented'
                                  .format(self.__class__.__name__,
                                          self.can_encode.__name__))

    @property
    def size(self):
        '''
        :return: The dimensionality of the vector space
                 used to represent tokens.
        '''
        raise NotImplementedError('abstract property {}.{} '
                                  'has not been implemented'
                                  .format(self.__class__.__name__,
                                          'size'))


class Word2VecEncoder(Encoder):
    '''
    Class representing the concept of
    an encoder from words to word embeddings.
    '''

    def __init__(self):
        self.model = None

    def fit(self, X, *args, **kwargs):
        '''
        Fits the model using the gensim Word2Vec behind the scenes.
        :param X: An iterable of sentences, each represented as a string.
        :param args: Forwarding parameters.
        :param kwargs: Forwarding parameters.
        '''
        self.model = Word2Vec([[w for w in s.split()] for s in X],
                              *args, **kwargs)

    def load(self, f_name, *args, **kwargs):
        '''
        Loads the parameters of a pre-trained model.
        :param f_name: The source path.
        :param args: Forwarding parameters.
        :param kwargs: Forwarding parameters.
        '''
        self.model = KeyedVectors.load(f_name, *args, **kwargs)

    def save(self, f_name, *args, **kwargs):
        '''
        Saves the parameters of the trained model.
        :param f_name: The target path.
        :param args: Forwarding parameters.
        :param kwargs: Forwarding parameters.
        '''
        self.model.save(f_name, *args, **kwargs)

    def encode(self, X):
        '''
        Encodes an iterable of words into an iterable of word vectors.
        :param X: An iterable of words, each represented as a string.
        :return: A numpy.ndarray of word embeddings.
        '''
        if self.model:
            return np.array([self.model.wv[x] for x in np.atleast_1d(X)])
        else:
            raise AttributeError('{}.model has not been initialized'
                             .format(self.__class__.__name__))

    def can_encode(self, seed):
        return seed in self.model.wv.vocab

    @property
    def size(self):
        '''
        :return: Returns the size of the vectors produced by
        the pre-trained Word2Vec tool.
        '''
        if self.model:
            return self.model.vector_size
        else:
            raise AttributeError('{}.model has not been initialized'
                             .format(self.__class__.__name__))


class OneHotEncoder(Encoder):
    '''
    Class representing an encoder from words to one-hot vectors.
    '''

    def __init__(self):
        self.index = dict()

    def fit(self, X):
        '''
        Fits the model.
        :param X: An iterable of words.
        '''
        self.index = {w:i for i, w in
                      enumerate(np.unique(np.atleast_1d(X)))}

    def encode(self, X):
        '''
        Encodes an iterable of words into an iterable of one-hot vectors.
        :param X: An iterable of words, each represented as a string.
        :return: A numpy.ndarray of one-hot vectors.
        '''
        res = []
        for w in np.atleast_1d(X):
            v = np.zeros(self.size, np.int32)
            try:
                v[self.index[w]] = 1
            except KeyError:
                warnings.warn('Word "{}" cannot be represented '
                              'in the given one-hot scheme'.format(w),
                              RuntimeWarning)
            res.append(v)
        return np.array(res)

    def can_encode(self, seed):
        return seed in self.index.keys()

    def save(self, f_name):
        pickle.dump(self.index, open(f_name, 'wb'))

    def load(self, f_name):
        self.index = pickle.load(open(f_name, 'rb'))

    @property
    def size(self):
        '''
        :return: The size of the vocabulary used to train the model.
        '''
        return len(self.index)


class MultiWordEncoder(Encoder):
    def __init__(self, word_encoder, topic_model, fn='add'):
        self.word_encoder = word_encoder
        self.topic_model = topic_model
        self.fn = fn

    def encode(self, X):
        topics = self.topic_model.predict(X)
        res = []
        for t in topics:
            vecs = []
            for w in t:
                if self.word_encoder.can_encode(w):
                    vecs.append(self.word_encoder.encode(w)[0])
            res.append(sum(vecs) if self.fn == 'add'
                       else sum(vecs) / len(vecs))
        return np.array(res)

    def can_encode(self, seed):
        for t in self.topic_model.tokenize(seed):
            if self.word_encoder.can_encode(t):
                return True
        return False

    @property
    def size(self):
        return self.word_encoder.size


class TopicModel():
    def __init__(self):
        self.model = None

    def fit(self, X, *args, **kwargs):
        texts = [self.tokenize(x) for x in X]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(t) for t in texts]
        self.model = LdaModel(corpus, id2word=dictionary, *args, **kwargs)

    def predict(self, X):
        res = []
        for x in np.atleast_1d(X):
            tokens = self.tokenize(x)
            topic = 0
            try:
                topic = self.model[self.model.id2word.doc2bow(tokens)][0][0]
            except IndexError as e:
                print('x: ' + str(x))
                print('tokens: ' + str(tokens))
            terms = [w for w, _ in self.model.get_topic_terms(topic)]
            words = [self.model.id2word.id2token[i] for i in terms]
            res.append(words)
        return res

    @staticmethod
    def tokenize(text):
        tokenizer = RegexpTokenizer(r'\w+')
        en_stop = get_stop_words('en')
        p_stemmer = PorterStemmer()
        raw = text.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [t for t in tokens if not t in en_stop]
        stemmed_tokens = [p_stemmer.stem(t) for t in stopped_tokens]
        return stemmed_tokens

    def save(self, f_name):
        self.model.save(f_name)

    def load(self, f_name):
        self.model = LdaModel.load(f_name)


class SentimentEncoder(Encoder):
    def __init__(self, word_encoder):
        self.word_encoder = word_encoder

    def encode(self, X):
        X = np.atleast_2d(X)
        return np.c_[X[:,0], self.word_encoder.encode(X[:,1])]

    def can_encode(self, seed):
        return self.word_encoder.can_encode(seed[1])

    @property
    def size(self):
        return self.word_encoder.size + 1





