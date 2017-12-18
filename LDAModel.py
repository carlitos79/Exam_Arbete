from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
import gensim
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

base_dir = "F:/No-Backup Zone/RNN_With_Embeddings/Non_Topic_Words/"

path_to_adjectives = base_dir + "Adjectives.txt"
adjectives = open(path_to_adjectives).read()
adjectives = adjectives.lower()

path_to_verbs = base_dir + "Verbs.txt"
verbs = open(path_to_verbs).read()
verbs = verbs.lower()

path_to_adverbs = base_dir + "Adverbs.txt"
adverbs = open(path_to_adverbs).read()
adverbs = adverbs.lower()

path_to_numbers = base_dir + "Numbers.txt"
numbers = open(path_to_numbers).read()
numbers = numbers.lower()

path_to_topics = base_dir + "Topics.txt"
topics = open(path_to_topics).read()
topics = topics.lower()

tokenizer = RegexpTokenizer(r'\w+')
en_stop = stopwords.words('english')
#s_stemmer = SnowballStemmer("english")

user_input = input("Enter string: ")

doc_set = [user_input]

texts = []

for i in doc_set:

    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    stopped_tokens = [i for i in tokens if not i in en_stop] # get rid of stop words
    #stemmed_tokens = [s_stemmer.stem(i) for i in stopped_tokens] # turn the word to its stem form
    #adjective_tokens = [i for i in stopped_tokens if not i in adjectives] # get rid of adjectives
    #verb_tokens = [i for i in adjective_tokens if not i in verbs]
    #adverb_tokens = [i for i in adjective_tokens if not i in adverbs]
    number_tokens = [i for i in stopped_tokens if not i in numbers]
    topic_tokens = [i for i in number_tokens if i in topics]

    texts.append(topic_tokens)

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
#result = ldamodel.show_topics(num_topics=1, num_words=2, formatted=False)

topic_words = []
for i in range(1):
    topic_terms = ldamodel.get_topic_terms(i,3)
    topic_words.append([dictionary[pair[0]] for pair in topic_terms])

for elem in topic_words:
    topic = ' '.join(elem)

print(topic)

#text = nltk.word_tokenize(topic)
#x = nltk.pos_tag(text)
#print(x)

#Those who improve with age embrace the power of personal growth and personal achievement and begin to replace youth with wisdom, innocence with understanding, and lack of purpose with self-actualization.