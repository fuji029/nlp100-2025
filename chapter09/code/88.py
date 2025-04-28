from transformers import BertForSequenceClassification, BertTokenizer
import torch
import os
import warnings

warnings.simplefilter('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    model_path = "output_SST/checkpoint-12630"
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    texts = [
        'The movie was full of fun.',
        'The movie was full of excitement.',
        'The movie was full of crap.',
        'The movie was full of rubbish.',
    ]

    encoded = [tokenizer(text, return_tensors="pt") for text in texts]

    for i, input in enumerate(encoded):
        output = model(**input)
        pred = torch.argmax(output.logits, dim=1).item()
        pred_label = "Positive" if pred == 1 else "Negative"
        print(f"text:{texts[i]}\tpred: {pred_label}")


if __name__ == "__main__":
    main()
