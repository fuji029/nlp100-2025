from google import genai
from google.genai.types import GenerateContentResponse
import os
import requests
import csv
from tqdm import tqdm
import time

# 問題データのダウンロード
url = "https://raw.githubusercontent.com/nlp-waseda/JMMLU/refs/heads/main/JMMLU/global_facts.csv"
filename = "data/global_facts.csv"
urlData = requests.get(url).content

with open(filename, mode="wb") as f:
    f.write(urlData)

with open(filename, "r", encoding="UTF-8") as f:
    data = csv.reader(f)
    data = [list(item) for item in data]

# 環境変数からAPIKEYを読み込み
client = genai.Client(api_key=os.environ["GEMINI_APIKEY"])


def preprocess(problemlist: list[str]) -> dict[str, str]:
    """
    csvから読み込んだ各行を選択肢の問題形式にする
    """
    problem = problemlist[0]
    choices = problemlist[1:5]
    answer = problemlist[5]
    content = (
        f"{problem}\n\n"
        f"A: {choices[0]}\n"
        f"B: {choices[1]}\n"
        f"C: {choices[2]}\n"
        f"D: {choices[3]}\n"
    )
    return {"content": content, "answer": answer}


problems = map(preprocess, data)
responses: list[GenerateContentResponse] = list()
accuracy = list()

# API制限を回避するために15回りクエストを送るごとに1分停止
count = 0

for problem in tqdm(problems):

    contents = (
        # 役割を与えることで精度向上を図る手法(Role-Play Prompting)
        # https://arxiv.org/abs/2308.07702
        "あなたは世界の様々な事実に詳しい専門家です。\n"
        "以下の世界事実に関する問いに答えなさい。出力は必ずA, B, C, Dの記号のみ簡潔に出力しなさい。\n\n"
        f"{problem['content']}"
    )

    response = client.models.generate_content(
        model="gemini-1.5-flash", contents=contents
    )

    response = response.text.replace("\n", "")

    responses.append(response)

    true_or_false = 1 if response == problem["answer"] else 0
    accuracy.append(true_or_false)

    count += 1
    if (count % 15 == 0):
        count = 0
        time.sleep(65)


with open("out/43.model.out", "w") as f:
    for text in responses:
        f.write(text + "\n")

acc = sum(accuracy) / len(accuracy)
print(f"Accuracy: {acc}\n")

with open("out/43.out", "w") as f:
    f.write(f"Accuracy: {acc}\n")
