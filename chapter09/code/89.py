import torch
import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer, BatchEncoding
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings

warnings.simplefilter('ignore')


def preprocess(example: dict[str, str | int], tokenizer: BertTokenizer, max_length: int = 128) -> BatchEncoding:
    encoded = tokenizer(
        example["sentence"], max_length=max_length, padding="max_length", truncation=True)
    encoded["labels"] = example["label"]
    return encoded


class MyModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(
            model_name, num_labels=num_labels)
        self.linear = torch.nn.Linear(768, 2)

    def forward(self, batch):
        batch_without_labels = {k: v.to("cuda:0") for k,
                                v in batch.items() if k != "labels"}
        output = self.bert(**batch_without_labels)
        mean = torch.mean(output.last_hidden_state, 1)
        output = self.linear(mean)
        return output

    def training_step(self, batch, batch_idx):
        # labelsを除いたものを取得
        batch_without_labels = {k: v for k,
                                v in batch.items() if k != "labels"}
        output = self.bert(**batch_without_labels)
        mean = torch.mean(output.last_hidden_state, 1)
        output = self.linear(mean)
        loss = F.cross_entropy(output, batch['labels'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_without_labels = {k: v for k,
                                v in batch.items() if k != "labels"}
        output = self.bert(**batch_without_labels)
        mean = torch.mean(output.last_hidden_state, 1)
        output = self.linear(mean)
        loss = F.cross_entropy(output, batch['labels'])
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    model_name = "google-bert/bert-base-uncased"
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = load_dataset(
        "stanfordnlp/sst2", split="train"
    ).with_format("torch")

    validation_dataset = load_dataset(
        "stanfordnlp/sst2", split="validation"
    ).with_format("torch")

    encoded_train_dataset = train_dataset.map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    encoded_validation_dataset = validation_dataset.map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=validation_dataset.column_names
    )

    train_dataloader = DataLoader(
        encoded_train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(
        encoded_validation_dataset, batch_size=256, shuffle=False)

    lr = 1e-5
    model = MyModel(model_name, 2, lr)

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        dirpath="MyModels"
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=3
    )

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=10,
        callbacks=[
            checkpoint, early_stopping
        ],
        devices=[0]
    )

    trainer.fit(model, train_dataloader, validation_dataloader)
    model = MyModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, num_labels=2, lr=lr)

    accuracy = []
    with torch.no_grad():
        for batch in validation_dataloader:
            outputs = model(batch)
            y = batch["labels"].to("cuda:0")
            preds = outputs.argmax(-1)
            num_correct = (preds == y).sum().item()
            acc = num_correct / y.size(0)
            accuracy.append(acc)
    print("validation accuracy: ", sum(accuracy) / len(accuracy))


if __name__ == "__main__":
    main()
