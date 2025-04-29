import evaluate
from pandas import DataFrame

perplexity = evaluate.load("perplexity", module_type="metric")
input_texts = [
    " The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises"
]

results = perplexity.compute(
    model_id="gpt2",
    add_start_token=False,
    predictions=input_texts
)

df = []
for text, ppl in zip(input_texts, results["perplexities"]):
    df.append({"text": text, "perplexity": ppl})

DataFrame(df).to_csv("out/93.tsv", sep="\t", index=False)
