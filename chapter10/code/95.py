from transformers import AutoTokenizer, pipeline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
access_token = os.environ["HUGGING_FACE_TOKEN"]

generator = pipeline(
    "text-generation", model="meta-llama/Llama-3.2-1B-Instruct", token=access_token)

chat = [
    {"role": "user", "content": "What do you call a sweet eaten after dinner?"},
    {"role": "assistant", "content": 'Is the answer "dessert"?'},
    {"role": "user", "content": "Please give me the plural form of the word with its spelling in reverse order."}
]

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", token=access_token)
prompt = tokenizer.apply_chat_template(chat, tokenize=False)
output = generator(prompt, max_new_tokens=40)

with open("out/95.md", "w") as f:
    f.write("Prompt:\n")
    f.write("```\n")
    f.write(prompt)
    f.write("\n```\n")
    f.write("Output:\n")
    f.write("```\n")
    f.write(output[0]["generated_text"])
    f.write("\n```\n")
