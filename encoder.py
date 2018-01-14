import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import numpy as np
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import Doc2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import TruncatedSVD

'''
Encoders:

Encoder                     - Base class for all concrete encoders
OneHotEncoder               - Encodes a word into a one-hot representation
Word2VecEncoder             - Encodes a word into a dense representation
SentenceEncoder             - Encodes a sentence into a dense representation.
SentenceSentimentEncoder    - Encodes a (sentence, sentiment) pair
                              into a dense representation.
'''


class Encoder():
    '''
    Abstract class representing the concept of an encoder.
    '''

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

    @property
    def size(self):
        '''
        :return: The dimensionality of the vector space
                 used to represent tokens.
        '''
        raise NotImplementedError('abstract property {}.{} '
                                  'has not been implemented'
                                  .format(self.__class__.__name__,
                                          'dimension'))


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
        raise AttributeError('{}.model has not been initialized'
                             .format(self.__class__.__name__))

    @property
    def vocab(self):
        return self.model.wv.vocab

    @property
    def size(self):
        '''
        :return: Returns the size of the vectors produced by
        the pre-trained Word2Vec tool.
        '''
        if self.model:
            return self.model.vector_size
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

    @property
    def vocab(self):
        return self.index

    @property
    def size(self):
        '''
        :return: The size of the vocabulary used to train the model.
        '''
        return len(self.index)


class SentenceEncoder(Encoder):
    '''
    Class representing an encoder from sentences to sentence embeddings.
    '''

    def __init__(self, word_encoder, a=0.0001):
        self.a = a
        self.word_encoder = word_encoder
        self.pc = None
        self.word_frequencies = None

    def fit(self, X):
        '''
        Fits the model using a pre-trained Word2Vec tool.
        :param X: An iterable of sentences, each represented as a string.
        :param f_name: Path to the pre-trained Word2Vec tool.
        '''
        sentences = [s.split(' ') for s in np.atleast_1d(X)]
        sentences = [[w for w in s if w in self.word_encoder.vocab]
                     for s in sentences]
        words = np.hstack(sentences)
        unique, counts = np.unique(words, return_counts=True)
        freqs = counts / len(words)
        self.word_frequencies = dict(zip(unique, freqs))
        VS = ([(1 / len(s))
               * sum([(self.a / self.a + self.word_frequencies[w])
                      * self.word_encoder.encode(w)[0] for w in s])
               for s in sentences])
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(VS)
        self.pc = svd.components_

    def encode(self, X):
        '''
        Encodes an iterable of se_.ntences into
        an iterable of sentence embeddings.
        :param X: An iterable of sentences, each represented as a string.
        :return: A numpy.ndarray of sentence embeddings.
        '''
        sentences = [[w for w in s.split(' ')
                      if w in self.word_encoder.vocab
                      and w in self.word_frequencies]
                     for s in np.atleast_1d(X)]
        VS = np.array([(1 / len(s))
              * sum([(self.a / self.a + self.word_frequencies[w])
                     * self.word_encoder.encode(w)[0] for w in s])
              for s in sentences])
        return VS - VS.dot(self.pc.T) * self.pc
        #return np.array([v - v.dot(self.pc.T) * self.pc for v in VS])

    @property
    def size(self):
        '''
        :return: Returns the size of the vectors produced by
        the pre-trained Word2Vec tool.
        '''
        return self.word_encoder.size


class SentenceSentimentEncoder(Encoder):
    '''
    Class representing an encoder from (sentence, sentiment) pairs
    to sentence embeddings.
    '''
    def __init__(self, sentence_encoder):
        self.sentence_encoder = sentence_encoder

    def fit(self, X):
        '''
        Fits the model
        :param X: An iterable of (sentence, sentiment) pairs, where
        each sentence is represented as a string and
        each sentiment is represented as an integer.
        :return: A numpy.ndarray of sentence embeddings.
        '''
        self.sentence_encoder.fit(np.hstack(np.atleast_2d(X)[:, :-1]))

    def encode(self, X):
        '''
        Encodes an iterable of (sentence, sentiment) pairs into
        an iterable of sentence embeddings.
        :param X: An iterable of (sentence, sentiment) pairs.
        :return: An iterable of sentence embeddings.
        '''
        X = np.atleast_2d(X)
        sentences = np.hstack(X[:, :-1])
        sentiments = X[:, -1].astype(np.int)
        return np.c_[self.sentence_encoder.encode(sentences),
                     sentiments]

    @property
    def size(self):
        '''
        :return: Returns the size of the vectors produced by
        the pre-trained Word2Vec tool.
        '''
        return self.sentence_encoder.size + 1


class Doc2VecEncoder(Encoder):
    def __init__(self, *args, **kwargs):
        self.model = Doc2Vec(*args, **kwargs)

    def fit(self, X):
        corpus = [TaggedDocument(simple_preprocess(x), [i])
                  for i,x in enumerate(np.atleast_1d(X))]
        self.model.build_vocab(corpus)
        self.model.train(corpus,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.iter)

    def encode(self, X):
        return np.array([self.model.infer_vector(simple_preprocess(x)) for x in np.atleast_1d(X)]) * 10

    def load(self, fname):
        self.model = KeyedVectors.load(fname)

    def save(self, fname):
        self.model.save(fname)

    @property
    def size(self):
        return self.model.vector_size



