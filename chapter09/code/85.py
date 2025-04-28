from transformers import BertTokenizer, BatchEncoding
from datasets import load_dataset


def preprocess(example: dict[str, str | int], tokenizer: BertTokenizer, max_length: int = 512) -> BatchEncoding:
    encoded = tokenizer(example["sentence"], max_length=max_length)
    encoded["labels"] = example["label"]
    return encoded


def main():
    model_name = "google-bert/bert-base-uncased"
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = load_dataset(
        "stanfordnlp/sst2", split="train"
    )

    validation_dataset = load_dataset(
        "stanfordnlp/sst2", split="validation"
    )

    encoded_train_dataset = train_dataset.map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    encoded_validation_dataset = validation_dataset.map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=validation_dataset.column_names
    )

    print(encoded_train_dataset[0])


if __name__ == "__main__":
    main()
