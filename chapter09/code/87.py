from transformers import BertForSequenceClassification, BertTokenizer, BatchEncoding, DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import numpy as np
from pprint import pprint
import os
import warnings

warnings.simplefilter('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def preprocess(example: dict[str, str | int], tokenizer: BertTokenizer, max_length: int = 128) -> BatchEncoding:
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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_label = train_dataset.features["label"]
    label2id = {label: id for id, label in enumerate(class_label.names)}
    id2label = {id: label for id, label in enumerate(class_label.names)}
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=class_label.num_classes,
        label2id=label2id,
        id2label=id2label
    )

    training_args = TrainingArguments(
        output_dir="output_SST",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=256,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        num_train_epochs=10,
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="accuracy",
        logging_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        logging_dir="logs"
    )

    def compute_accuracy(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == labels).mean()}

    trainer = Trainer(
        model,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_validation_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_accuracy,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    eval_metrics = trainer.evaluate(encoded_validation_dataset)
    pprint(eval_metrics)


if __name__ == "__main__":
    main()
