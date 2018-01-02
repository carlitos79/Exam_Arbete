import codecs
from encoder import Word2VecEncoder
from encoder import OneHotEncoder
from encoder import SentenceEncoder
from decoder import Decoder

# Test

seeds = []
targets = []
start_seq = '\t'
end_seq = '\n'
with codecs.open('quotes.txt', encoding='utf-8') as f:
    lines = f.read().lower().replace('\r', '')\
        .replace('.', '').replace('!', '')\
        .replace('?','').replace(',', '').split('\n')
for line in lines:
    s, t = line.split(';;')
    seeds.append(s)
    t = '\t' + t + '\n'
    targets.append(t)
n_tokens = len(set([c for c in ''.join(targets)]))
max_seq_length = max([len(txt) for txt in targets])

# Example OneHotEncoder
'''
encoder = OneHotEncoder()
encoder.fit(seeds)
'''

#Example Word2VecEncoder
'''
encoder = Word2VecEncoder()
encoder.load('pretrained.wv')
'''

# Example SentenceEncoder
we = Word2VecEncoder()
we.load('pretrained.wv')
encoder = SentenceEncoder(we)
encoder.fit([t[1:-1] for t in targets])
print([t[1:-1] for t in targets])

decoder = Decoder(encoder, n_tokens, max_seq_length)
decoder.fit([t[1:-1] for t in targets], targets, batch_size=32, epochs=70)
test_seed = 'my life is great'
test_target = decoder.decode([test_seed])[0]
print('seed:', test_seed)
print('target:', test_target)