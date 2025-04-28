from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    input_text = "The movie was full of"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]

    logits, pred_token_id = torch.topk(next_token_logits, k=10)
    logits = logits.numpy()[0]
    pred_token_id = pred_token_id.numpy()[0]
    predicted_tokens = [tokenizer.decode(token_id)
                        for token_id in pred_token_id]

    for token, logit in zip(predicted_tokens, logits):
        print(f"{token.replace(' ', '')}: {logit}")


if __name__ == "__main__":
    main()
