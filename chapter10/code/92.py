from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")

text = "The movie was full of"

inputs = tokenizer(text, return_tensors='pt').to("cuda:0")
input_len = inputs["input_ids"].size(1)
output = model.generate(**inputs, max_new_tokens=40,
                        return_dict_in_generate=True, output_scores=True)

ids = output.sequences
scores = output.scores

tokens = tokenizer.convert_ids_to_tokens(ids[0])
tokens = [word.replace("Ġ", "") for word in tokens]
logits = [score[0][id].item() for score, id in zip(scores, ids[0])]

result = []

for token, logit in zip(tokens[input_len:], logits):
    result.append({"token": token, "logit": logit})

pd.DataFrame(result).to_csv("out/92.tsv", sep="\t", index=False)
