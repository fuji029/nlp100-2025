import gensim

model: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(
    "data/GoogleNews-vectors-negative300.bin", binary=True)

# 国名のファイル(https://gist.github.com/kalinchernev/486393efcca01623b18d)
with open("data/countries.txt", "r") as f:
    countries = f.read().split("\n")

for i, country in enumerate(countries):
    countries[i] = country.replace(" ", "_")
