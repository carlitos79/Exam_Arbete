from ende_test import test_word2vec
from ende_test import test_onehot

test_word2vec(seeds=['age, amazing', 'best'],
              source_path='data/Ranked_Quotes.txt',
              target_path='results.txt')


test_onehot(seeds=['age', 'amazing', 'best'],
            source_path='data/Ranked_Quotes.txt',
            target_path='results.txt')


