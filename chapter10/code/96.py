from transformers import AutoTokenizer, pipeline
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
access_token = os.environ["HUGGING_FACE_TOKEN"]

generator = pipeline(
    "text-generation", model="meta-llama/Llama-3.2-1B-Instruct", token=access_token)


def getChat(content: str):
    return [
        {"role": "system", "content": "Determine if this statement is positive or negative. You must output only \"positive\" or \"negative\"."},
        {"role": "user", "content": content},
    ]


tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", token=access_token)

with open("../chapter08/data/SST-2/dev.tsv") as f:
    data = f.read().rstrip().split("\n")
    data = [(item.split("\t")) for item in data[1:]]

acc = []
for sentence, label in tqdm(data):
    prompt = tokenizer.apply_chat_template(getChat(sentence), tokenize=False)
    output = generator(prompt, max_new_tokens=40)[
        0]["generated_text"].split("<|eot_id|>")[2].replace("\n", "").replace("\s\S", "").split()[-1]
    label = "positive" if label == "1" else "negative"
    acc.append(1 if label in output else 0)

print(f"dev acc: {sum(acc)/len(acc)}")
