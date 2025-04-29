from transformers import pipeline, TextGenerationPipeline
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

generator: TextGenerationPipeline = pipeline(model="openai-community/gpt2")
result = []
for temperature in range(2, 10, 2):
    generated_text = generator(
        "The movie was full of",
        temperature=temperature / 10)[0]['generated_text'].replace('\n', ' ')
    result.append({"temperature": temperature/10,
                  "generated_text": generated_text})

pd.DataFrame(result).to_csv("out/91.tsv", sep="\t")
