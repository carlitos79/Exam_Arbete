from nltk.corpus import sentiwordnet
from Utils import *
import nltk

################################ METHOD #############################
def MaxOf3(pos, neg, obj):

    max = pos
    if neg > max:
        max = neg
    if obj > max:
        max = obj
    if neg > obj:
        max = neg
    return max

############################## METHOD ###############################
def GetMorpho(word):

    morphos = list(sentiwordnet.senti_synsets(word))
    morphos_to_str = ''.join(str(morph) for morph in morphos)

    word_len = len( word ) + 1
    index = morphos_to_str.index(word)

    morpho = ''.join(str(morph) for morph in morphos_to_str[index + word_len])

    return morpho

############################ METHOD ####################################
def RankQuote(quote):

    pos = 0
    neg = 0
    obj = 0

    quote_without_stops = text_to_wordlist(quote, remove_stopwords=True, stem_words=False)
    tokenized_quote = nltk.word_tokenize(quote_without_stops)
    #print(tokenized_quote)

    for word in tokenized_quote:

        morph = GetMorpho(word)
        #print(morph)

        input_word = sentiwordnet.senti_synset(word + '.' + morph + '.01')

        positive = input_word.pos_score()
        negative = input_word.neg_score()
        objective = input_word.obj_score()

        #print("Positive: ", positive)
        #print("Negative: ", negative)
        #print("Objective: ", objective)
        #print( "\n" )
        #print( "Local max: ", MaxOf3(positive, negative, objective))
        #print("\n")

        pos = positive + pos
        neg = negative + neg
        obj = objective + obj

    final_max = MaxOf3(pos, neg, obj)

    if final_max == pos:
        return "positive"
    if final_max == neg:
        return "negative"
    if final_max == obj:
        return "objective"