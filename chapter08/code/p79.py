from p70 import emb
from p71 import train, dev
from p75 import collate_fn
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import argparse


class SST2Dataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx].int(), self.labels[idx]


class MLPModel(nn.Module):
    def __init__(self, embed_dim, embedding, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids):
        emb = self.embedding(input_ids)   # (B, L, D)
        h = emb.mean(dim=1)               # (B, D)
        logits = self.mlp(h)              # (B, 1)
        return torch.sigmoid(logits).squeeze(-1)


class CNNModel(nn.Module):
    def __init__(self, embed_dim, embedding, num_filters=100, kernel_size=3):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.linear = nn.Linear(num_filters, 1)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)        # (B, L, D)
        emb = emb.transpose(1, 2)              # (B, D, L)
        h = self.conv(emb)                     # (B, F, L)
        h = h.max(dim=2).values                # Global max pooling → (B, F)
        logits = self.linear(h)                # (B, 1)
        return torch.sigmoid(logits).squeeze(-1)


class RNNModel(nn.Module):
    def __init__(self, embed_dim, embedding, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)      # (B, L, D)
        _, h_n = self.rnn(emb)                # h_n: (1, B, H)
        h = h_n.squeeze(0)                    # (B, H)
        logits = self.linear(h)               # (B, 1)
        return torch.sigmoid(logits).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, embed_dim, embedding, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)       # (B, L, D)
        _, (h_n, _) = self.lstm(emb)           # h_n: (1, B, H)
        h = h_n.squeeze(0)                     # (B, H)
        h = self.dropout(h)
        logits = self.linear(h)                # (B, 1)
        return torch.sigmoid(logits).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-model", choices=("MLP", "CNN", "RNN", "LSTM"), required=True)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-num_epochs", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=64)
    args = parser.parse_args()
    print(vars(args))

    device = f"cuda:{args.device}"
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    train_dataset = SST2Dataset([item["input_ids"] for item in train], [item["label"] for item in train])
    dev_dataset = SST2Dataset([item["input_ids"] for item in dev], [item["label"] for item in dev])

    train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_batch = DataLoader(dev_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

    models = [MLPModel(emb.size()[1], emb), CNNModel(emb.size()[1], emb), RNNModel(emb.size()[1], emb), LSTMModel(emb.size()[1], emb)]
    model_name = ["MLP", "CNN", "RNN", "LSTM"]
    print(f"======={args.model}=======")
    model = models[model_name.index(args.model)]

    model.to(device)
    model.train()

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_hist = [0] * num_epochs
    epoch_times = []

    for epoch in range(num_epochs):
        start_time = time.perf_counter()

        for input_ids, labels in tqdm(train_batch, desc=f"epoch {epoch:2}"):
            pred = model(input_ids.to(device))
            loss = loss_fn(pred, labels.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist[epoch] += loss.item()

        end_time = time.perf_counter()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

        print(f"epoch: {epoch} loss: {loss_hist[epoch] / len(train_dataset) * batch_size:.4f}")
        if epoch > 1 and loss_hist[epoch] >= loss_hist[epoch - 1]:
            print(f"Early Stopping at epoch {epoch}")
            break
        if epoch == num_epochs - 1:
            print(f"Finished at epoch {epoch}")

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"Average time per epoch: {avg_epoch_time:.2f}s")

    dev_acc = 0
    model.eval()
    with torch.no_grad():
        for input_ids, labels in dev_batch:
            preds = model(input_ids.to(device))
            preds = (preds > 0.5).float()
            dev_acc += (preds == labels.to(device)).sum().item()
    print(f"dev accuracy: {dev_acc/len(dev_dataset)}")


if __name__ == "__main__":
    main()
