import gensim
from gensim.models import KeyedVectors
import re
from tqdm import tqdm

model: KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(
    "data/GoogleNews-vectors-negative300.bin", binary=True)

# データを読み込み
pattern = re.compile(r"\: [0-9a-zA-Z\-]+\n")
with open("data/questions-words.txt", "r") as f:
    data = f.read().rstrip()

titles: list[str] = pattern.findall(data)
all_words = pattern.split(data)[1:]
all_words = [item.rstrip().split("\n") for item in all_words]

all_words = [list(map((lambda x: x.split(" ", maxsplit=3)), item))
             for item in all_words]
title_and_words = [{"title": title.rstrip(), "words": words}
                   for title, words in zip(titles, all_words)]


def print_similality(words: list[str], title: str) -> None:
    results = model.most_similar(
        positive=[words[1], words[2]], negative=[words[0]])[0]
    print(f"{title},{','.join(words)},{results[0]},{results[1]}")


for item in tqdm(title_and_words):
    for words in tqdm(item["words"]):
        print_similality(words, item["title"])
