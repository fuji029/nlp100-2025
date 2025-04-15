import gensim
from gensim.models import KeyedVectors

model: KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(
    "data/GoogleNews-vectors-negative300.bin", binary=True)

results = model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"])
for result in results:
    print(result)
