import codecs
import string
from encoder import Word2VecEncoder
from encoder import OneHotEncoder
from decoder import Decoder
import time

# Cmon git!!!


def load_data(f_name):
    '''
    Loads the data in the specified file.
    :param f_name: The filename of the data location.
    :return: Three lists: sentiments, topics, quotes
    '''
    print('Reading data from file...')
    with codecs.open(f_name, encoding='utf-8') as f:
        lines = f.read().split('\n')
    sentiments, topics, quotes = zip(*[(s,t,q) for s,t,q in
                                       [l.split(';;') for l in lines]])
    print('...Finished reading data from file')
    print('Pre-processing data...')
    sentiments = [1 if s == 'positive'
                  else -1 if s == 'negative' else 0
                  for s in sentiments]
    table = str.maketrans('', '', string.punctuation)
    quotes = [' '.join([w.lower().translate(table) for w in q.split()])
              for q in quotes]
    print('...Finished pre-processing data')
    return sentiments, topics, quotes


def write_result(model, train_time, test_time, seeds, results, f_name):
    '''
    Writes the result of a test.
    :param model: String representation of the model that was tested.
    :param seeds: The seeds that was tested on.
    :param results: The resulting text sequences that was obtained.
    :param f_name: The filename which to write the results to.
    :return:
    '''
    print('Writing results to file...')
    seed_space = max(len(s) for s in seeds)
    form_str = '{:' + str(seed_space) + 's} --> {}\n'
    with open(f_name, 'at+') as f:
        f.write('Results from decoder(' + model + '):\n\n')
        f.write('Training time: {}\n'.format(train_time))
        f.write('Testing time: {}\n\n'.format(test_time))
        for s, r in zip(seeds, results):
            f.write(form_str.format(s, r))
        f.write('\n' + '-'*100 + '\n')
    print('...Finished writing results to file')


def test_word2vec(seeds, source_path, target_path):
    '''
    Tests a decoder model with a word2vec encoder.
    :param seeds: The seeds to test the model on.
    :param source_path: The path to the training data.
    :param target_path: The path to the results data.
    :return:
    '''
    _, topics, quotes = load_data(source_path)
    print('Loading pre-trained word2vec model...')
    encoder = Word2VecEncoder()
    encoder.load('pretrained.wv')
    print('...Finished loading pre-trained word2vec model')
    print('Initializing the decoder...')
    decoder = Decoder(encoder)
    print('...Finished initializing the decoder')
    print('Fitting the decoder...')
    t = time.time()
    decoder.fit(topics, quotes, epochs=25, trace=True)
    train_time = time.time() - t
    print('...Finished fitting the decoder')
    print('Predicting sequences...')
    t = time.time()
    results = decoder.decode(seeds)
    test_time = time.time() - t
    print('...Finished predicting sequences')
    write_result('word2vec', train_time, test_time, seeds, results, target_path)


def test_onehot(seeds, source_path, target_path):
    '''
    Tests a decoder model with a one-hot encoder.
    :param seeds: The seeds to test the model on.
    :param source_path: The path to the training data.
    :param target_path: The path to the results data.
    :return:
    '''
    _, topics, quotes = load_data(source_path)
    print('Fitting the one-hot encoder...')
    encoder = OneHotEncoder()
    encoder.fit(topics)
    print('...Finished fitting the one-hot encoder')
    print('Initializing the decoder...')
    decoder = Decoder(encoder)
    print('...Finished initializing the decoder')
    print('Fitting the decoder...')
    t = time.time()
    decoder.fit(topics, quotes, epochs=25, trace=True)
    train_time = time.time() - t
    print('...Finished fitting the decoder')
    print('Predicting sequences...')
    t = time.time()
    results = decoder.decode(seeds)
    test_time = time.time() - t
    print('...Finished predicting sequences')
    write_result('one-hot', train_time, test_time, seeds, results, target_path)

test_word2vec(seeds=['age', 'amazing', 'architecture', 'attitude'],
              source_path='data/Ranked_Quotes.txt',
              target_path='results.txt')

test_onehot(seeds=['age', 'amazing', 'architecture', 'attitude'],
            source_path='data/Ranked_Quotes.txt',
            target_path='results.txt')

