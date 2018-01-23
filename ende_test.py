import string
import codecs
from encoder import Word2VecEncoder
from encoder import OneHotEncoder
from decoder import Decoder

CHECK_MARK = u'\u2713'


def load_data(f_name, n_samples):
    '''
    Loads the data in the specified file.
    :param f_name: The filename of the data location.
    :return: Three lists: sentiments, topics, quotes
    '''
    print('Reading data from file...', end='')
    with codecs.open(f_name, encoding='utf-8') as f:
        lines = f.read().split('\n')
    if n_samples != 'all':
        lines = lines[:n_samples]
    sentiments, topics, quotes = zip(*[(s,t,q) for s,t,q in
                                       [l.split(';;') for l in lines]])
    print(CHECK_MARK)
    print('Pre-processing data...', end='')
    sentiments = [1 if s == 'positive'
                  else -1 if s == 'negative' else 0
                  for s in sentiments]
    table = str.maketrans('', '', string.punctuation)
    quotes = [' '.join([w.lower().translate(table) for w in q.split()])
              for q in quotes]
    print(CHECK_MARK)
    return sentiments, topics, quotes


def write_result(model, train_time, test_time, seeds, results, f_name):
    '''
    Writes the result of a test.
    :param model: String representation of the model that was tested.
    :param seeds: The seeds that was tested on.
    :param results: The resulting text sequences that was obtained.
    :param f_name: The filename which to write the results to.
    '''
    print('Writing results to file...', end='')
    seed_space = max(len(s) for s in seeds)
    form_str = '{:' + str(seed_space) + 's} --> {}\n'
    with open(f_name, 'at+') as f:
        f.write('Results from decoder(' + model + '):\n\n')
        f.write('Training time: {}\n'.format(train_time))
        f.write('Testing time: {}\n\n'.format(test_time))
        for s, r in zip(seeds, results):
            f.write(form_str.format(s, r))
        f.write('\n' + '-'*100 + '\n')
    print(CHECK_MARK)


def test_word2vec(source_path, target_paths, n_samples):
    '''
    Tests a decoder model with a word2vec encoder.
    :param seeds: The seeds to test the model on.
    :param source_path: The path to the training data.
    :param target_path: The paths to save the model to.
    '''
    _, topics, quotes = load_data(source_path, n_samples)
    print('Loading pre-trained word2vec model...', end='')
    encoder = Word2VecEncoder()
    encoder.load('pretrained.wv')
    print(CHECK_MARK)
    print('Initializing the decoder...', end='')
    decoder = Decoder(encoder)
    print(CHECK_MARK)
    print('Fitting the decoder...')
    decoder.fit_generator(topics, quotes, epochs=25, batch_size=32, trace=True)
    print('Fitting the decoder...' + CHECK_MARK)
    print('Saving model...', end='')
    decoder.save(target_paths)
    print(CHECK_MARK)


def test_onehot(source_path, target_paths, n_samples):
    '''
    Tests a decoder model with a one-hot encoder.
    :param seeds: The seeds to test the model on.
    :param source_path: The path to the training data.
    :param target_path: The paths to save the model to.
    '''
    _, topics, quotes = load_data(source_path, n_samples)
    print('Fitting the one-hot encoder...', end='')
    encoder = OneHotEncoder()
    encoder.fit(topics)
    print(CHECK_MARK)
    print('Initializing the decoder...', end='')
    decoder = Decoder(encoder)
    print(CHECK_MARK)
    print('Fitting the decoder...')
    decoder.fit_generator(topics, quotes, epochs=25, batch_size=32, trace=True)
    print('Fitting the decoder...' + CHECK_MARK)
    print('Saving model...', end='')
    decoder.save(target_paths)
    print(CHECK_MARK)
