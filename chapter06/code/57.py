import gensim
from sklearn.cluster import KMeans

model: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(
    "data/GoogleNews-vectors-negative300.bin", binary=True)

# 国名のファイル(https://gist.github.com/kalinchernev/486393efcca01623b18d)
with open("data/countries.txt", "r") as f:
    countries = f.read().split("\n")

for i, country in enumerate(countries):
    countries[i] = country.replace(" ", "_")

countries = [country for country in countries if country in model.key_to_index]
country_vectors = [model[country] for country in countries]

kmeans_model = KMeans(n_clusters=5)
kmeans_model.fit(country_vectors)

result = {i: [] for i in range(5)}
for country, label in zip(countries, kmeans_model.labels_):
    result[label].append(country)

for k, v in result.items():
    print(f"label:{k}")
    for country in v:
        print(f"\t{country}")
    print(f"total count: {len(v)}\n")
