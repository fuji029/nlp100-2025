from transformers import pipeline, TextGenerationPipeline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

generator: TextGenerationPipeline = pipeline(model="openai-community/gpt2")
print("temperature\tgenerated_text")
for temperature in range(2, 10, 2):
    outputs = generator(
        "The movie was full of",
        temperature=temperature / 10)[0]['generated_text'].replace('\n', ' ')
    print(f"{temperature/10}\t{outputs}")
