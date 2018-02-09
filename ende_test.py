import os
import string
import codecs
import pickle
from collections import defaultdict
from encoder import Word2VecEncoder
from encoder import OneHotEncoder
from encoder import MultiWordEncoder
from encoder import TopicModel
from encoder import SentimentEncoder
from decoder import Decoder
from itertools import chain


CHECK_MARK = u'\u2713'


def load_data(f_name, sample_size):
    '''
    Loads data from the specified file.
    :param f_name: The source file of the data.
    :param sample_size: The percentage of samples to draw.
    :return: Three tuples: sentiments, topics & quotes.
    '''
    print('Reading data from file...', end='', flush=True)
    with codecs.open(f_name, encoding='utf-8') as f:
        lines = f.read().split('\n')
    lines = lines[:9116] + lines[9117:]
    sentiments, topics, quotes = zip(*[(s,t,q) for s,t,q in
                                       [l.split(';;') for l in lines]])
    print(CHECK_MARK, flush=True)

    print('Pre-processing data...', end='', flush=True)
    sentiments = [1 if s == 'positive'
                  else -1 if s == 'negative' else 0
                  for s in sentiments]
    table = str.maketrans('', '', string.punctuation)
    quotes = [' '.join([w.lower().translate(table) for w in q.split()])
              for q in quotes]
    print(CHECK_MARK, flush=True)

    print('Sampling data...', end='', flush=True)
    data_dict = defaultdict(list)
    for t, q, s in zip(topics, quotes, sentiments):
        data_dict[t].append((s, t, q))
    data_set = []
    for k in data_dict.keys():
        for e in data_dict[k][:int(len(data_dict[k]) * sample_size)]:
            data_set.append(e)
    print(CHECK_MARK, flush=True)

    return zip(*data_set)

"""
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
"""


def test_word2vec(sample_size,
                  epochs,
                  batch_size,
                  data_path,
                  word2vec_filename,
                  dec_filenames):

    '''
    Tests a decoder based on a word2vec encoder.
    :param sample_size: The percentage of data to train on.
    :param epochs: The number of epochs to train on.
    :param batch_size: The size of each batch.
    :param data_path: The path to the data.
    :param word2vec_filename: Source or target for the word2vec resource.
    :param dec_filenames: Two filenames (root, weights) for the decoder resource.
    '''
    _, topics, quotes = load_data(data_path, sample_size)

    encoder = Word2VecEncoder()
    if os.path.isfile(word2vec_filename):
        print('Loading the word2vec encoder...', end='', flush=True)
        encoder.load(word2vec_filename)
        print(CHECK_MARK, flush=True)
    else:
        print('Fitting the word2vec encoder...', end='', flush=True)
        encoder.fit(topics)
        print(CHECK_MARK, flush=True)

    print('Initializing the decoder...', end='', flush=True)
    decoder = Decoder(encoder)
    print(CHECK_MARK)
    print('Fitting the decoder...', flush=True)
    decoder.fit_generator(topics,
                          quotes,
                          epochs=epochs,
                          batch_size=batch_size,
                          trace=True)
    print('Fitting the decoder...' + CHECK_MARK, flush=True)

    print('Saving the decoder...', end='', flush=True)
    decoder.save(dec_filenames)
    print(CHECK_MARK, flush=True)

    if not os.path.isfile(word2vec_filename):
        print('Saving the encoder...', end='', flush=True)
        encoder.save(word2vec_filename)
        print(CHECK_MARK, flush=True)


def test_onehot(sample_size,
                epochs,
                batch_size,
                data_path,
                onehot_filename,
                dec_filenames):
    '''
    Tests a decoder based on a word2vec encoder.
    :param sample_size: The percentage of data to train on.
    :param epochs: The number of epochs to train on.
    :param batch_size: The size of each batch.
    :param data_path: The path to the data.
    :param onehot_filename: Source or target for the one-hot resource.
    :param dec_filenames: Two filenames (root, weights) for the decoder resource.
    '''
    _, topics, quotes = load_data(data_path, sample_size)

    encoder = OneHotEncoder()
    if os.path.isfile(onehot_filename):
        print('Loading the one-hot encoder...', end='', flush=True)
        encoder.load(onehot_filename)
        print(CHECK_MARK, flush=True)
    else:
        print('Fitting the one-hot encoder...', end='', flush=True)
        encoder.fit(topics)
        print(CHECK_MARK, flush=True)

    print('Initializing the decoder...', end='', flush=True)
    decoder = Decoder(encoder)
    print(CHECK_MARK, flush=True)
    print('Fitting the decoder...', flush=True)
    decoder.fit_generator(topics,
                          quotes,
                          epochs=epochs,
                          batch_size=batch_size,
                          trace=True)
    print('Fitting the decoder...' + CHECK_MARK, flush=True)

    print('Saving the decoder...', end='', flush=True)
    decoder.save(dec_filenames)
    print(CHECK_MARK, flush=True)

    if not os.path.isfile(onehot_filename):
        print('Saving the one-hot encoder...', end='', flush=True)
        with open(onehot_filename, 'wb') as f:
            pickle.dump(encoder, f)
        print(CHECK_MARK, flush=True)


def test_multihot(sample_size,
                  epochs,
                  batch_size,
                  data_path,
                  onehot_filename,
                  topic_filename,
                  dec_filenames):
    '''
    Tests a decoder based on a word2vec encoder.
    :param sample_size: The percentage of data to train on.
    :param epochs: The number of epochs to train on.
    :param batch_size: The size of each batch.
    :param data_path: The path to the data.
    :param onehot_filename: Source or target for the one-hot resource.
    :param topic_filename: Source or target for the topic model resource.
    :param dec_filenames: Two filenames (root, weights) for the decoder resource.
    '''
    _, _, quotes = load_data(data_path, sample_size)

    topic_model = TopicModel()
    if os.path.isfile(topic_filename):
        print('Loading the topic model...', end='', flush=True)
        topic_model.load(topic_filename)
        print(CHECK_MARK, flush=True)
    else:
        print('Fitting the topic model...', end='', flush=True)
        topic_model.fit(quotes)
        print(CHECK_MARK, flush=True)

    onehot_encoder = OneHotEncoder()
    if os.path.isfile(onehot_filename):
        print('Loading the one-hot encoder...', end='', flush=True)
        onehot_encoder.load(onehot_filename)
        print(CHECK_MARK, flush=True)
    else:
        print('Fitting the one-hot encoder...', end='', flush=True)
        onehot_encoder.fit(list(set(chain(*[topic_model.tokenize(q) for q in quotes]))))
        print(CHECK_MARK, flush=True)

    print('Initializing the multi-hot encoder...', end='', flush=True)
    encoder = MultiWordEncoder(onehot_encoder, topic_model, fn='add')
    print(CHECK_MARK, flush=True)

    print('Initializing the decoder...', end='', flush=True)
    decoder = Decoder(encoder)
    print(CHECK_MARK, flush=True)
    print('Fitting the decoder...', flush=True)
    decoder.fit_generator(quotes,
                          quotes,
                          epochs=epochs,
                          batch_size=batch_size,
                          trace=True)
    print('Fitting the decoder...' + CHECK_MARK, flush=True)

    print('Saving the decoder...', end='', flush=True)
    decoder.save(dec_filenames)
    print(CHECK_MARK, flush=True)

    if not os.path.isfile(onehot_filename):
        print('Saving the one-hot encoder...', end='', flush=True)
        with open(onehot_filename, 'wb') as f:
            pickle.dump(onehot_encoder, f)
        print(CHECK_MARK, flush=True)

    if not os.path.isfile(topic_filename):
        print('Saving topic model...', end='', flush=True)
        topic_model.save(topic_filename)
        print(CHECK_MARK, flush=True)


def test_multiword2vec(sample_size,
                       epochs,
                       batch_size,
                       data_path,
                       word2vec_filename,
                       topic_filename,
                       dec_filenames):
    '''
    Tests a decoder based on a word2vec encoder.
    :param sample_size: The percentage of data to train on.
    :param epochs: The number of epochs to train on.
    :param batch_size: The size of each batch.
    :param data_path: The path to the data.
    :param word2vec_filename: Source or target for the word2vec resource.
    :param topic_filename: Source or target for the topic model resource.
    :param dec_filenames: Two filenames (root, weights) for the decoder resource.
    '''
    _, _, quotes = load_data(data_path, sample_size)

    topic_model = TopicModel()
    if os.path.isfile(topic_filename):
        print('Loading the topic model...', end='', flush=True)
        topic_model.load(topic_filename)
        print(CHECK_MARK, flush=True)
    else:
        print('Fitting the topic model...', end='', flush=True)
        topic_model.fit(quotes)
        print(CHECK_MARK, flush=True)

    word2vec_encoder = Word2VecEncoder()
    if os.path.isfile(word2vec_filename):
        print('Loading the word2vec encoder...', end='', flush=True)
        word2vec_encoder.load(word2vec_filename)
        print(CHECK_MARK, flush=True)
    else:
        print('Fitting the word2vec encoder...', end='', flush=True)
        word2vec_encoder.fit([topic_model.tokenize(q) for q in quotes])
        print(CHECK_MARK, flush=True)

    print('Initializing the multi-word2vec encoder...', end='', flush=True)
    encoder = MultiWordEncoder(word2vec_encoder, topic_model, fn='average')
    print(CHECK_MARK, flush=True)

    print('Initializing the decoder...', end='', flush=True)
    decoder = Decoder(encoder)
    print(CHECK_MARK, flush=True)
    print('Fitting the decoder...', flush=True)
    decoder.fit_generator(quotes,
                          quotes,
                          epochs=epochs,
                          batch_size=batch_size,
                          trace=True)
    print('Fitting the decoder...' + CHECK_MARK, flush=True)

    print('Saving the decoder...', end='', flush=True)
    decoder.save(dec_filenames)
    print(CHECK_MARK, flush=True)

    if not os.path.isfile(word2vec_filename):
        print('Saving the word2vec encoder...', end='', flush=True)
        with open(word2vec_filename, 'wb') as f:
            pickle.dump(word2vec_encoder, f)
        print(CHECK_MARK, flush=True)

    if not os.path.isfile(topic_filename):
        print('Saving topic model...', end='', flush=True)
        topic_model.save(topic_filename)
        print(CHECK_MARK, flush=True)


def test_sentiment(sample_size,
                   epochs,
                   batch_size,
                   data_path,
                   word2vec_filename,
                   dec_filenames):
    sentiments, topics, quotes = load_data(data_path, sample_size)

    w2v = Word2VecEncoder()
    if os.path.isfile(word2vec_filename):
        print('Loading the word2vec encoder...', end='', flush=True)
        w2v.load(word2vec_filename)
        print(CHECK_MARK, flush=True)
    else:
        print('Fitting the word2vec encoder...', end='', flush=True)
        w2v.fit(topics)
        print(CHECK_MARK, flush=True)
    encoder = SentimentEncoder(w2v)

    print('Initializing the decoder...', end='', flush=True)
    decoder = Decoder(encoder)
    print(CHECK_MARK)
    print('Fitting the decoder...', flush=True)
    decoder.fit_generator([z for z in zip(sentiments, topics)],
                          quotes,
                          epochs=epochs,
                          batch_size=batch_size,
                          trace=True)
    print('Fitting the decoder...' + CHECK_MARK, flush=True)

    print('Saving the decoder...', end='', flush=True)
    decoder.save(dec_filenames)
    print(CHECK_MARK, flush=True)

    if not os.path.isfile(word2vec_filename):
        print('Saving the encoder...', end='', flush=True)
        w2v.save(word2vec_filename)
        print(CHECK_MARK, flush=True)


def test_sentiment2(sample_size,
                   epochs,
                   batch_size,
                   data_path,
                   oh_filename,
                   dec_filenames):
    sentiments, topics, quotes = load_data(data_path, sample_size)

    oh = OneHotEncoder()
    if os.path.isfile(oh_filename):
        print('Loading the oh encoder...', end='', flush=True)
        oh = pickle.load(open(oh_filename, 'rb'))
        print(CHECK_MARK, flush=True)
    else:
        print('Fitting the oh encoder...', end='', flush=True)
        oh.fit(topics)
        print(CHECK_MARK, flush=True)
    encoder = SentimentEncoder(oh)

    print('Initializing the decoder...', end='', flush=True)
    decoder = Decoder(encoder)
    print(CHECK_MARK)
    print('Fitting the decoder...', flush=True)
    decoder.fit_generator([z for z in zip(sentiments, topics)],
                          quotes,
                          epochs=epochs,
                          batch_size=batch_size,
                          trace=True)
    print('Fitting the decoder...' + CHECK_MARK, flush=True)

    print('Saving the decoder...', end='', flush=True)
    decoder.save(dec_filenames)
    print(CHECK_MARK, flush=True)

    if not os.path.isfile(oh):
        print('Saving the encoder...', end='', flush=True)
        pickle.dump(oh, open(oh_filename, 'wb'))
        print(CHECK_MARK, flush=True)

