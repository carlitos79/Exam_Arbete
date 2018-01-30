from ende_test import test_word2vec
from ende_test import test_onehot
from ende_test import test_multihot
from ende_test import test_multiword2vec

test_word2vec(sample_size=0.05,
              epochs=5,
              batch_size=32,
              data_path='data/Ranked_Quotes.txt',
              word2vec_filename='pretrained.wv',
              dec_filenames=['w2v_root.bin', 'w2v_weights.bin'])

test_onehot(sample_size=0.5,
            epochs=5,
            batch_size=32,
            data_path='data/Ranked_Quotes.txt',
            onehot_filename='pretrained_oh.bin',
            dec_filenames=['oh_root.bin', 'oh_weights.bin'])

test_multiword2vec(sample_size=0.5,
                   epochs=5,
                   batch_size=32,
                   data_path='data/Ranked_Quotes.txt',
                   word2vec_filename='pretrained.wv',
                   topic_filename='pretrained_topic.bin',
                   dec_filenames=['mw2v_root.bin', 'mw2v_weights.bin'])

test_multihot(sample_size=0.5,
              epochs=5,
              batch_size=32,
              data_path='data/Ranked_Quotes.txt',
              onehot_filename='pretrained_mh.bin',
              topic_filename='pretrained_topic.bin',
              dec_filenames=['mh_root.bin', 'mh_weights.bin'])






