import pandas as pd
from p70 import token2id
import torch

train_df = pd.read_csv("data/SST-2/train.tsv", sep="\t")
dev_df = pd.read_csv("data/SST-2/dev.tsv", sep="\t")


def load_data(df: pd.DataFrame):
    texts = df["sentence"]
    labels = df["label"]
    res = []

    for text, label in zip(texts, labels):
        input_ids = torch.tensor([token2id[w]
                                  for w in text.split() if w in token2id])
        if len(input_ids) == 0:
            continue
        d = {
            "text": text,
            "label": torch.tensor(label, dtype=torch.float32),
            "input_ids": input_ids
        }
        res.append(d)

    return res


train = load_data(train_df)
dev = load_data(dev_df)

if __name__ == "__main__":
    from pprint import pprint
    pprint(train[1])
