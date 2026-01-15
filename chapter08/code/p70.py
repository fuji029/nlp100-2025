import gensim
import numpy as np
from torch import tensor

w2vmodel: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(
    "../chapter06/data/GoogleNews-vectors-negative300.bin", binary=True)
emb = w2vmodel.vectors
emb = np.concatenate([np.zeros((1, 300), dtype=np.float32), emb])
emb = tensor(emb)

id2token = ["<PAD>"] + w2vmodel.index_to_key
token2id = w2vmodel.key_to_index
token2id = {"<PAD>": 0} | {key: token2id[key] + 1 for key in token2id}

if __name__ == "__main__":
    print(token2id["<PAD>"], id2token.index("<PAD>"))
    print(token2id["Japan"], id2token.index("Japan"))
