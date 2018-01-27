from ende_test import test_word2vec
from ende_test import test_onehot
from ende_test import test_multihot
from ende_test import test_multiword2vec

test_word2vec(n_samples = 100,
              source_path='data/Ranked_Quotes.txt',
              target_paths=['w2v_root.bin', 'w2v_weights.bin'])

"""
test_onehot(n_samples = 100,
              source_path='data/Ranked_Quotes.txt',
              target_paths=['oh_root.bin', 'oh_weights.bin'])

test_multihot(n_samples = 100,
              source_path='data/Ranked_Quotes.txt',
              target_paths=['mh_root.bin', 'mh_weights.bin'])

test_multiword2vec(n_samples = 100,
              source_path='data/Ranked_Quotes.txt',
              target_paths=['mw2v_root.bin', 'mw2v_weights.bin'])
"""






