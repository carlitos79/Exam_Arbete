import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Encoder

class Encoder():
    def fit(self, X, *args, **kwargs):
        raise NotImplementedError('abstract function {}.{}'
                                  '(X, *args, **kwargs) '
                                  'has not been implemented'
                                  .format(self.__class__.__name__,
                                          self.fit.__name__))

    def encode(self, X):
        raise NotImplementedError('abstract function {}.{}'
                                  '(X) has not been implemented'
                                  .format(self.__class__.__name__,
                                          self.encode.__name__))

    def save(self, f_name, *args, **kwargs):
        raise NotImplementedError('abstract function {}.{}'
                                  '(f_name, *args, **kwargs) '
                                  'has not been implemented'
                                  .format(self.__class__.__name__,
                                          self.save.__name__))

    def load(self, f_name, format, *args, **kwargs):
        raise NotImplementedError('abstract function {}.{}'
                                  '(f_name, format, *args, **kwargs) '
                                  'has not been implemented'
                                  .format(self.__class__.__name__,
                                          self.load.__name__))

    @property
    def dimension(self):
        raise NotImplementedError('abstract property {}.{} '
                                  'has not been implemented'
                                  .format(self.__class__.__name__,
                                          'dimension'))


class Word2VecEncoder(Encoder):
    def __init__(self):
        self.model = None

    def fit(self, X, *args, **kwargs):
        self.model = Word2Vec([s.split(' ') for s in X], *args, **kwargs)

    def encode(self, X):
        if self.model:
            return np.array([self.model.wv[x] for x in np.atleast_2d(X)])
        raise AttributeError('{}.model has not been initialized'
                             .format(self.__class__.__name__))

    def save(self, f_name, *args, **kwargs):
        if self.model:
            self.model.save(f_name, *args, **kwargs)
        else:
            raise AttributeError('{}.model has not been initialized'
                                 .format(self.__class__.__name__))

    def load(self, f_name, format=False, *args, **kwargs):
        if format:
            self.model = KeyedVectors.load_word2vec_format(f_name, binary=True)
        else:
            self.model = KeyedVectors.load(f_name, *args, **kwargs)

    @property
    def dimension(self):
        if self.model:
            return self.model.vector_size
        raise AttributeError('{}.model has not been initialized'
                             .format(self.__class__.__name__))