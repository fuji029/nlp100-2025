import gensim
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(
    "data/GoogleNews-vectors-negative300.bin", binary=True)

with open("data/countries.txt", "r") as f:
    countries = f.read().split("\n")

for i, country in enumerate(countries):
    countries[i] = country.replace(" ", "_")

countries = [country for country in countries if country in model.key_to_index]
country_vectors = np.array([np.array(model[country]) for country in countries])

tsne = TSNE(n_components=2, random_state=42)
res = tsne.fit_transform(country_vectors)

clustered = KMeans(n_clusters=5).fit_predict(country_vectors)

# 参考: https://qiita.com/FukuharaYohei/items/bef37174397f8c6ef8e7
plt.figure(figsize=(10, 10))
cmap = plt.get_cmap('Dark2')
plt.scatter(res[:, 0], res[:, 1], marker='.', c=[
            cmap(clustered[i] / 4) for i in range(len(clustered))])
for country, re, cl in zip(countries, res, clustered):
    plt.annotate(country, xy=(re[0], re[1]), c=(cmap(cl / 4)))
plt.savefig("out/59.png")
