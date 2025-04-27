from transformers import pipeline
from pandas import DataFrame

model = "google-bert/bert-base-uncased"
fill_mask = pipeline(
    "fill-mask", model=model, device=1, top_k=10
)

masked_text = "The movie was full of [MASK]."

outputs = fill_mask(masked_text)
print(DataFrame(outputs[:10]))
