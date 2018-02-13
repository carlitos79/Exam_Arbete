import pickle
from decoder import Decoder
from encoder import Word2VecEncoder
from encoder import SentimentEncoder

# test word2vec encoder
enc = Word2VecEncoder()
enc.load('pretrained.wv')
dec = Decoder.load(['w2v_root.bin', 'w2v_weights.bin'], enc)
print(dec.decode('age'))

# test one-hot encoder
enc = pickle.load(open('pretrained_oh.bin', 'rb'))
dec = Decoder.load(['oh_root.bin', 'oh_weights.bin'], enc)
print(dec.decode('age'))

# test word2vec + sentiment encoder
w2v = Word2VecEncoder()
w2v.load('pretrained.wv')
enc = SentimentEncoder(w2v)
dec = Decoder.load(['sent_root_w2v.bin', 'sent_weights_w2v.bin'], enc)
print(dec.decode([(1, 'age')]))

# test one-hot + sentiment encoder
oh = pickle.load(open('pretrained_oh.bin', 'rb'))
enc = SentimentEncoder(oh)
dec = Decoder.load(['sent_root_oh.bin', 'sent_weights_oh.bin'], enc)
print(dec.decode([(1, 'age')]))



