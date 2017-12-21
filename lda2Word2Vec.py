from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
import gensim
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors

####################################### DIRECTORIES ##########################################

base_dir = "F:/No-Backup Zone/RNN_With_Embeddings/Non_Topic_Words/"

path_to_numbers = base_dir + "Numbers.txt"
numbers = open(path_to_numbers).read()
numbers = numbers.lower()

path_to_topics = base_dir + "Topics.txt"
topics = open(path_to_topics).read()
topics = topics.lower()

tokenizer = RegexpTokenizer(r'\w+')
en_stop = stopwords.words('english')

####################################### LOAD MODEL ###########################################
google_corpus_path = "F:/No-Backup Zone/RNN_With_Embeddings/word2vec_model"
google_model = KeyedVectors.load_word2vec_format(google_corpus_path)

######################################## METHOD... ###########################################
def lda2w2vInput(input_quote):

    user_input = input_quote
    doc_set = [user_input]
    texts = []

    for i in doc_set:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        number_tokens = [i for i in stopped_tokens if not i in numbers]
        topic_tokens = [i for i in number_tokens if i in topics]
        texts.append(topic_tokens)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)

    topic_words = []
    for i in range(1):
        topic_terms = ldamodel.get_topic_terms(i,3)
        topic_words.append([dictionary[pair[0]] for pair in topic_terms])

    for elem in topic_words:
        topic = ' '.join(elem)

    return topic

######################################## METHOD... ###########################################
def getContextWord(input_quote):

    matching_words = []

    key_words = lda2w2vInput(input_quote)
    no_match = google_model.wv.doesnt_match(key_words.split())
    single_words = [word for word in key_words.split()]

    for word in single_words:
        if len(single_words) > 1:
            if word != no_match:
                matching_words.append(word.lower())
                last_match = google_model.wv.doesnt_match(matching_words)

            return last_match

        else:
            single_word = ''.join(str(one_word) for one_word in single_words)
            return single_word
