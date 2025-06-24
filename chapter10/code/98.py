import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    BatchEncoding,
)
from datasets import load_dataset
import numpy as np
from pprint import pprint
import os
import warnings

warnings.simplefilter('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_prompt(input: str, label: int) -> str:
    label = "positive" if label == 1 else "negative"
    prompt = f"Determine if this statement is positive or negative. You must output only \"positive\" or \"negative\".\nstatement: {input}\nanswer: {label}"
    return prompt


def preprocess(example: dict[str, str | int], tokenizer, max_length: int = 128) -> BatchEncoding:
    sentences = [make_prompt(sentence, label) for sentence, label in zip(example["sentence"], example["label"])]
    encoded = tokenizer(
        sentences,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    encoded["labels"] = tokenizer(
        sentences,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"]
    return encoded


def main():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset(
        "stanfordnlp/sst2", split="train"
    )

    validation_dataset = load_dataset(
        "stanfordnlp/sst2", split="validation"
    )

    encoded_train_dataset = train_dataset.map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=train_dataset.column_names,
        batched=True,
        batch_size=64
    )
    encoded_validation_dataset = validation_dataset.map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=validation_dataset.column_names,
        batched=True,
        batch_size=8
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir="output_SST/p98",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=8,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        num_train_epochs=3,
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="accuracy",
        logging_strategy="epoch",
        eval_strategy="epoch",
        eval_accumulation_steps=10,
        load_best_model_at_end=True,
        fp16=True,
        logging_dir="logs",
    )

    def compute_accuracy(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=-1)
        preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
        labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

        preds = np.array([1 if "positive" in pred.split(":")[-1].lower() else 0 for pred in preds])
        labels = np.array([1 if "positive" in label.split(":")[-1].lower()
                           else 0 for label in labels])
        return {"accuracy": (preds == labels).mean()}

    trainer = Trainer(
        model,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_validation_dataset,
        args=training_args,
        compute_metrics=compute_accuracy,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    pprint(eval_metrics)


if __name__ == "__main__":
    main()
