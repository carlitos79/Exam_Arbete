from ende_test import test_word2vec
from ende_test import test_onehot

test_word2vec(source_path='data/Ranked_Quotes.txt',
              target_paths=['w2c_root.bin', 'w2v_weights.bin'],
              n_samples='all')


test_onehot(source_path='data/Ranked_Quotes.txt',
            target_paths=['oh_root.bin', 'oh_weights.bin'],
            n_samples='all')


