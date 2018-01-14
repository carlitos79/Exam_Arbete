from nltk.corpus import sentiwordnet
from nltk.stem.wordnet import WordNetLemmatizer
from Utils import *
import nltk

verbs_path = "F:/No-Backup Zone/RNN_With_Embeddings/Non_Topic_Words/Verbs2.txt"
verbs = open(verbs_path).read()

conjunctions_path = "F:/No-Backup Zone/RNN_With_Embeddings/Non_Topic_Words/Conjunctions.txt"
conjunctions = open(conjunctions_path).read()

################################ METHOD ###################################
def MaxOf3(pos, neg, obj):

    max = pos
    if neg > max:
        max = neg
    if obj > max:
        max = obj
    if neg > obj:
        max = neg
    return max

################################# METHOD ####################################
def GetMorpho(word):

    morphos = list(sentiwordnet.senti_synsets(word))
    morphos_to_str = ''.join(str(morph) for morph in morphos)

    word_len = len(word) + 1
    index = morphos_to_str.index(word)

    morpho = ''.join(str(morph) for morph in morphos_to_str[index + word_len])

    return morpho

################################### METHOD ########################################
def RankQuote(quote):

    pos = 0
    neg = 0
    obj = 0

    lemmatizer = WordNetLemmatizer()

    ################################## Filtering #########################################
    quote_without_stops = text_to_wordlist(quote, remove_stopwords=True, stem_words=False)
    #print(quote_without_stops)
    tokenized_quote = nltk.word_tokenize(quote_without_stops)
    #print(tokenized_quote)

    for word in tokenized_quote:
        if word in conjunctions:
            tokenized_quote.remove(word)

    for word in tokenized_quote:
        if word in verbs:
            infinitive = lemmatizer.lemmatize(word, "v")
            tokenized_quote.append(infinitive)
        tokenized_quote.remove(word)
    print(tokenized_quote)
    #######################################################################################

    ################################ Quote classification #################################
    for word in tokenized_quote:

        try:
            morph = GetMorpho(word)
            input_word = sentiwordnet.senti_synset(word + '.' + morph + '.01')

            positive = input_word.pos_score()
            negative = input_word.neg_score()
            objective = input_word.obj_score()

            #print("Positive: ", positive)
            #print("Negative: ", negative)
            #print("Objective: ", objective)
            #print( "LOCAL MAX: ", MaxOf3(positive, negative, objective))
            #print("\n")

            pos = positive + pos
            neg = negative + neg
            obj = objective + obj

        except:
            pass

    final_max = MaxOf3(pos, neg, obj)

    if final_max == pos:
        return "positive"
    if final_max == neg:
        return "negative"
    if final_max == obj:
        return "objective"