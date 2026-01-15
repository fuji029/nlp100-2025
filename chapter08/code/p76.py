from p70 import emb
from p71 import train, dev
from p72 import BoWModel
from p75 import collate_fn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


class SST2Dataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx].int(), self.labels[idx]


def main():
    lr = 1e-3
    num_epochs = 100
    batch_size = 64

    model = BoWModel(embed_dim=emb.size()[1], embedding=emb)

    train_dataset = SST2Dataset([item["input_ids"] for item in train], [item["label"] for item in train])
    dev_dataset = SST2Dataset([item["input_ids"] for item in dev], [item["label"] for item in dev])

    train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_batch = DataLoader(dev_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model.to("cuda:0")
    model.train()

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_hist = [0] * num_epochs
    epoch_times = []

    for epoch in range(num_epochs):
        start_time = time.perf_counter()

        for input_ids, labels in tqdm(train_batch, desc=f"epoch {epoch:2}"):
            pred = model(input_ids.to("cuda:0"))
            loss = loss_fn(pred, labels.to("cuda:0"))
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
            preds = model(input_ids.to("cuda:0"))
            preds = (preds > 0.5).float()
            dev_acc += (preds == labels.to("cuda:0")).sum().item()
    print(f"dev accuracy: {dev_acc/len(dev_dataset)}")


if __name__ == "__main__":
    main()
