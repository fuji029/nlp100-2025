from google import genai
import os

# 環境変数からAPIKEYを読み込み
client = genai.Client(api_key=os.environ["GEMINI_APIKEY"])

with open("./data/neko.txt", "r") as f:
    content = f.read()

tokens = client.models.count_tokens(
    model="gemini-1.5-flash", contents=content
)

print(tokens)
