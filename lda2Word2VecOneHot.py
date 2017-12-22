from nltk.tokenize import RegexpTokenizer
from gensim import corpora
import gensim
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors

####################################### DIRECTORIES ##########################################

base_dir = "F:/No-Backup Zone/RNN_With_Embeddings/Non_Topic_Words/"

path_to_numbers = base_dir + "Numbers.txt"
numbers_list = open( path_to_numbers ).read()
numbers_list = numbers_list.lower()

path_to_topics = base_dir + "Topics.txt"
topics_list = open( path_to_topics ).read()
topics_list = topics_list.lower()

tokenizer = RegexpTokenizer(r'\w+')
en_stop = stopwords.words('english')

####################################### LOAD MODEL ###########################################
google_corpus_path = "F:/No-Backup Zone/RNN_With_Embeddings/word2vec_model"
google_model = KeyedVectors.load_word2vec_format(google_corpus_path)

######################################## METHOD... ###########################################
def Input2OneHotEncoder(input_quote):

    quote = [input_quote]
    topics = []

    for i in quote:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        number_tokens = [i for i in stopped_tokens if not i in numbers_list]
        topic_tokens = [i for i in number_tokens if i in topics_list]
        topics.append(topic_tokens)

    topic_words = []
    dictionary = corpora.Dictionary(topics)
    corpus = [dictionary.doc2bow(text) for text in topics]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)

    for i in range(1):
        topic_terms = ldamodel.get_topic_terms(i,3)
        topic_words.append([dictionary[pair[0]] for pair in topic_terms])

    for word in topic_words:
        topic = ' '.join(word)
    return topic

######################################## METHOD... ###########################################
def Input2Word2VecEncoder(input_quote):

    matching_words = []
    key_words = Input2OneHotEncoder(input_quote)
    #print("Result from lda: " + key_words)
    no_match = google_model.wv.doesnt_match(key_words.split())
    single_words = [word for word in key_words.split()]

    for word in single_words:
        if len(single_words) > 1:
            if word != no_match:
                matching_words.append(word.lower())
                context_word = matching_words[0]
                return context_word
        else:
            context_word = ''.join(str(one_word) for one_word in single_words)
            return context_word
