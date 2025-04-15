import gensim
from gensim.models import KeyedVectors
import pandas as pd
from scipy.stats import spearmanr

model: KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(
    "data/GoogleNews-vectors-negative300.bin", binary=True)

df = pd.read_csv("data/wordsim353/combined.csv",
                 names=["word1", "word2", "human"], header=0)


cosSims = list()
for index, item in df.iterrows():
    cosSim = model.similarity(item["word1"], item["word2"])
    cosSims.append(cosSim)

corr, _ = spearmanr(cosSims, df["human"])
print(corr)
