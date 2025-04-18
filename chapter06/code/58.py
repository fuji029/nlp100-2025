import gensim
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

model: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(
    "data/GoogleNews-vectors-negative300.bin", binary=True)

with open("data/countries.txt", "r") as f:
    countries = f.read().split("\n")

for i, country in enumerate(countries):
    countries[i] = country.replace(" ", "_")

countries = [country for country in countries if country in model.key_to_index]
country_vectors = [model[country] for country in countries]

# 参考: https://qiita.com/FukuharaYohei/items/8648da8bbad27c841479
clustered = linkage(country_vectors, method="ward")

plt.figure(figsize=(20, 50), dpi=100)
_ = dendrogram(clustered, labels=(countries),
               leaf_font_size=8, orientation="right")
plt.savefig("out/58.png")
