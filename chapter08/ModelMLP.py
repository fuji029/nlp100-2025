import torch
import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchinfo


class MyDataSet:
    def __init__(self, model) -> None:
        vocab: dict = model.key_to_index
        for key in vocab:
            vocab[key] += 1
        vocab["<SEP>"] = 0
        self.vocab = vocab

    def __call__(self, data: pd.DataFrame):
        texts = data["sentence"]
        labels = data["label"]
        result: list[dict] = []
        for text, label in zip(texts, labels):
            d = {}
            d["text"] = text
            d["label"] = torch.tensor(label)
            d["input_ids"] = [self.vocab[w]
                              for w in text.split() if w in self.vocab]
            if (len(d["input_ids"]) == 0):
                continue
            result.append(d)
        return result


class MyDataLoader:
    def __init__(self, batch_size: int, model) -> None:
        self.batch_size = batch_size
        self.model = model
        emb = model.vectors
        emb = np.concatenate([np.zeros(((1, 300)), dtype=float), emb])
        self.emb = emb
        self.vocab_size = len(model.key_to_index) + 1

    def __call__(self, dataset: list[dict]):
        x_batch = []
        y_batch = []
        batch = []

        for i, item in tqdm(enumerate(dataset), total=len(dataset)):
            x = item["input_ids"]
            y = item["label"]
            x_batch.append(x)
            y_batch.append(y)
            if ((i+1) % self.batch_size == 0):
                x_batch = self.collate(x_batch)
                batch.append(
                    [torch.tensor(np.array(x_batch)), torch.tensor(np.array(y_batch))])
                x_batch = []
                y_batch = []
        if (len(x_batch)):
            x_batch = self.collate(x_batch)
            batch.append(
                [torch.tensor(np.array(x_batch)), torch.tensor(np.array(y_batch))])
        return batch

    def collate(self, x_batch: list):
        max_len = max([len(input_ids) for input_ids in x_batch])
        x_batch = torch.tensor(np.array([
            input_ids + [0 for _ in range(max_len - len(input_ids))] for input_ids in x_batch], dtype=int))
        return x_batch


class MyModel(torch.nn.Module):
    def __init__(self, emb, vocab_size):
        super(MyModel, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, 300)
        self.embedding.weight = torch.nn.Parameter(
            torch.from_numpy(emb))
        self.linear = torch.nn.Linear(300, 150)
        self.hidden1 = torch.nn.Linear(150, 150)
        self.hidden2 = torch.nn.Linear(150, 150)
        self.hidden3 = torch.nn.Linear(150, 2)
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.embedding(x)
        x = torch.mean(x, 1)
        x = self.linear(x)
        x = self.activation(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.softmax(x)
        return x


def main():

    train_df = pd.read_csv("data/SST-2/train.tsv", sep="\t")
    dev_df = pd.read_csv("data/SST-2/dev.tsv", sep="\t")
    w2vmodel: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(
        "../chapter06/data/GoogleNews-vectors-negative300.bin", binary=True)
    emb = w2vmodel.vectors
    emb = np.concatenate([np.zeros((1, 300), dtype=np.float32), emb])

    dataset = MyDataSet(model=w2vmodel)
    train_dataset = dataset(train_df)
    dev_dataset = dataset(dev_df)

    train_dataloader = MyDataLoader(batch_size=32, model=w2vmodel)
    dev_dataloader = MyDataLoader(batch_size=128, model=w2vmodel)

    train_batch = train_dataloader(train_dataset)
    dev_batch = dev_dataloader(dev_dataset)

    vocab_size = len(w2vmodel.key_to_index) + 1
    model = MyModel(emb, vocab_size)
    model.to("cuda:0")
    lr = 1e-4
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 20
    loss_hist = [0] * num_epochs
    for epoch in tqdm(range(num_epochs)):
        for x_batch, y_batch in train_batch:
            pred = model(x_batch.to("cuda:0"))
            loss = loss_fn(pred, y_batch.squeeze_().to("cuda:0"))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist[epoch] += loss.item()
        tqdm.write(
            f"epoch: {epoch} loss: {loss_hist[epoch] / len(train_batch):.4f}")
    dev_acc = 0
    with torch.no_grad():
        for x_batch, y_batch in dev_batch:
            pred = model(x_batch.to("cuda:0"))
            dev_acc += (torch.argmax(pred, dim=1) ==
                        y_batch.squeeze_().to("cuda:0")).sum()
    print(f"dev accuracy: {dev_acc/len(dev_df)}")


if __name__ == "__main__":
    main()
