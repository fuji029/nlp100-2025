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
    cls_embs = [model(**inputs).last_hidden_state[:, 0, :]
                for inputs in tokens]

cos_sim = torch.nn.CosineSimilarity()
for i, cls in enumerate(cls_embs):
    for j in range(i+1, len(texts)):
        sim = cos_sim(cls, cls_embs[j]).item()
        print(f"{texts[i]}\t{texts[j]}\t{sim:.4f}")
