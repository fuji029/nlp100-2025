from transformers import BertModel, BertTokenizer
import torch

texts = [
    'The movie was full of fun.',
    'The movie was full of excitement.',
    'The movie was full of crap.',
    'The movie was full of rubbish.',
]

model_name = "google-bert/bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)

tokens = [tokenizer(text, return_tensors="pt") for text in texts]
with torch.no_grad():
    last_hidden_states = [
        model(**inputs).last_hidden_state for inputs in tokens]
    mean_embs = [torch.mean(last_hidden_state, 1)
                 for last_hidden_state in last_hidden_states]

cos_sim = torch.nn.CosineSimilarity()
for i, cls in enumerate(mean_embs):
    for j in range(i+1, len(texts)):
        sim = cos_sim(cls, mean_embs[j]).item()
        print(f"{texts[i]}\t{texts[j]}\t{sim:.4f}")
