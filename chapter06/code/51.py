import gensim

model = gensim.models.KeyedVectors.load_word2vec_format(
    "data/GoogleNews-vectors-negative300.bin", binary=True)

print(model.similarity("United_States", "U.S."))
