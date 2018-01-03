from Sentiment_Classification import *

def RankQuotesFile(recipient_file, source_file):

    source = open(source_file, "r" )

    with open(recipient_file, "w") as text_file:

        for line in source:

            single_quote = line.split()

            quote_2_string = ' '.join(str(quote) for quote in single_quote)

            semi = quote_2_string.index(";;")

            item_to_classify = quote_2_string[(semi + 2):]

            translation = str.maketrans("","", string.punctuation);

            print(item_to_classify.translate(translation))

            item_to_classify = item_to_classify.translate(translation)

            sentiment = RankQuote(item_to_classify)

            ranked_quote = sentiment + ";;" + quote_2_string

            text_file.write("%s\n" % ranked_quote)

        source.close()

target = "F:/No-Backup Zone/RNN_With_Embeddings/Data/Ranked_Quotes.txt"
source = "F:/No-Backup Zone/RNN_With_Embeddings/Data/quotes3.txt"

RankQuotesFile(target, source)

