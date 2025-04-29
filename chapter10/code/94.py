from transformers import AutoTokenizer, pipeline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
access_token = os.environ["HUGGING_FACE_TOKEN"]

generator = pipeline(
    "text-generation", model="meta-llama/Llama-3.2-1B-Instruct", token=access_token)

chat = [
    {"role": "user", "content": "What do you call a sweet eaten after dinner?"}
]

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", token=access_token)
prompt = tokenizer.apply_chat_template(chat, tokenize=False)
print(prompt.replace("\n", " "))

output = generator(prompt)
print(output[0]["generated_text"].replace("\n", " "))
