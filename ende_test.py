import codecs
from encoder import Word2VecEncoder
from decoder import Decoder

# Test

seeds = []
targets = []
start_seq = '\t'
end_seq = '\n'
with codecs.open('quotes.txt', encoding='utf-8') as f:
    lines = f.read().lower().split('\n')
for line in lines:
    s, t = line.split(';;')
    seeds.append(s)
    t = '\t' + t + '\n'
    targets.append(t)
n_tokens = len(set([c for c in ''.join(targets)]))
max_seq_length = max([len(txt) for txt in targets])

# Set the path of the google pre-trained word2vec file
path = 'GoogleNews-vectors-negative300.bin'
encoder = Word2VecEncoder()
encoder.load(path, format=True)
decoder = Decoder(encoder, n_tokens, max_seq_length)
decoder.fit(seeds, targets, batch_size=32, epochs=70)

test_seed = 'life'
test_target = decoder.decode([test_seed])[0]
print('seed:', test_seed)
print('target:', test_target)