from peft import LoraConfig, TaskType
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
from datasets import load_dataset
from pprint import pprint
import os
import warnings

warnings.simplefilter('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def make_prompt(input: str) -> str:
    prompt = f"Determine if this statement is positive or negative. You must output only \"positive\" or \"negative\".\nstatement: {input}\nanswer: "
    return prompt


def convert_to_dpo_format(example: dict, tokenizer) -> dict:
    def get_label(label: int) -> str:
        return "positive" if label == 1 else "negative"

    prompt = make_prompt(example["sentence"])
    chosen = get_label(example["label"]) + tokenizer.eos_token
    rejected = get_label((example["label"]+1) % 2) + tokenizer.eos_token
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main():
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset(
        "stanfordnlp/sst2", split="train"
    )

    validation_dataset = load_dataset(
        "stanfordnlp/sst2", split="validation"
    )

    encoded_train_dataset = train_dataset.map(
        lambda x: convert_to_dpo_format(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    encoded_validation_dataset = validation_dataset.map(
        lambda x: convert_to_dpo_format(x, tokenizer),
        remove_columns=validation_dataset.column_names
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
    )

    peft_config = LoraConfig(
        r=128,
        lora_alpha=128,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj"],
    )

    dpo_config = DPOConfig(
        output_dir="output_SST/p99",
        bf16=True,
        max_steps=100,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        beta=0.1,
        max_prompt_length=512,
        max_length=1024
    )

    dpo_trainer = DPOTrainer(
        model,
        args=dpo_config,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_validation_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    dpo_trainer.train()

    eval_metrics = dpo_trainer.evaluate()
    pprint(eval_metrics)


if __name__ == '__main__':
    main()
