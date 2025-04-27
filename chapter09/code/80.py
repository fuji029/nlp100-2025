from transformers import AutoTokenizer, BertTokenizer

model = "google-bert/bert-base-uncased"
sentence = "The movie was full of incomprehensibilities."

tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model)
tokens = tokenizer.tokenize(sentence)
print(tokens)
