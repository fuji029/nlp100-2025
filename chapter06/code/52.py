import gensim

model = gensim.models.KeyedVectors.load_word2vec_format(
    "data/GoogleNews-vectors-negative300.bin", binary=True)

results = model.most_similar(positive=["United_States"])
for result in results:
    print(result)
