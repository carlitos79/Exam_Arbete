import codecs
import string
from encoder import Word2VecEncoder
from encoder import OneHotEncoder
from decoder import Decoder

# Cmon git!!!


def load_data(f_name):
    '''
    Loads the data in the specified file.
    :param f_name: The filename of the data location.
    :return: three lists: sentiments, topics, quotes
    '''
    print('Reading data from file...')
    with codecs.open(f_name, 'rt', encoding='utf-8') as f:
        lines = f.read().split('\n')
    sentiments, topics, quotes = zip(*[(s,t,q) for s,t,q in
                                       [l.split(';;') for l in lines]])
    print('Preprocessing data...')
    sentiments = [1 if s == 'positive'
                  else -1 if s == 'negative' else 0
                  for s in sentiments]
    table = str.maketrans('', '', string.punctuation)
    quotes = [' '.join([w.lower().translate(table) for w in q.split()])
              for q in quotes]
    return sentiments, topics, quotes


def write_result(model, seeds, results, f_name):
    '''
    Writes the result of a test.
    :param model: String representation of the model that was tested.
    :param seeds: The seeds that was tested on.
    :param results: The resulting text sequences that was obtained.
    :param f_name: The filename which to write the results.
    :return:
    '''
    seed_space = max(len(s) for s in seeds)
    with open(f_name, 'at+') as f:
        f.write('Results from decoder(' + model + '):\n\n')
        for s, r in zip(seeds, results):
            f.write('{:' + str(seed_space) + 's} --> {}\n'.format(s, r))
        f.write('\n' + '-'*100 + '\n')


def test_word2vec_decoder(seeds, source_path, target_path):
    '''
    Tests a decoder model with a word2vec encoder.
    :param seeds: The seeds to test the model on.
    :param source_path: The path to the training data.
    :param target_path: The path to the results data.
    :return:
    '''
    _, topics, quotes = load_data(source_path)
    encoder = Word2VecEncoder()
    encoder.load('pretrained.wv')
    decoder = Decoder(encoder)
    decoder.fit(topics, quotes, epochs=1, trace=True)
    results = decoder.decode(seeds)
    write_result('word2vec', seeds, results, target_path)


def save_onehot_decoder(seeds, source_path, target_path):
    '''
    Tests a decoder model with a one-hot encoder.
    :param seeds: The seeds to test the model on.
    :param source_path: The path to the training data.
    :param target_path: The path to the results data.
    :return:
    '''
    _, topics, quotes = load_data(source_path)
    encoder = OneHotEncoder()
    encoder.fit(topics)
    decoder = Decoder(encoder)
    decoder.fit(topics, quotes, trace=True)
    results = decoder.decode(seeds)
    write_result('one-hot', seeds, results, target_path)

test_word2vec_decoder(seeds=['age', 'amazing', 'best'],
                      source_path='data/quotes3.txt',
                      target_path='results.txt')

