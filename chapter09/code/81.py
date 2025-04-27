from transformers import pipeline

model = "google-bert/bert-base-uncased"
fill_mask = pipeline(
    "fill-mask", model=model, device=1
)

masked_text = "The movie was full of [MASK]."

outputs = fill_mask(masked_text)
print("token:", outputs[0]["token_str"])
print("sequence:", outputs[0]["sequence"])
