import pandas as pd

with open("out/54.out", "r") as f:
    df = pd.read_csv(f, header=None)

sem = df[~df[0].str.contains("gram")]
syn = df[df[0].str.contains("gram")]

print("Acc of semantic: ", (sem[4] == sem[5]).sum() / len(sem))
print("Acc of syntactic: ", (syn[4] == syn[5]).sum() / len(syn))
