from google import genai
import os

# 環境変数からAPIKEYを読み込み
client = genai.Client(api_key=os.environ["GEMINI_APIKEY"])

contents = (
    "あなたは日本の歌人です。\n"
    "春の季節にあった川柳を10個作ってください。作った川柳をリスト形式で出力して\n"
)

response = client.models.generate_content(
    model="gemini-1.5-flash", contents=contents,
)
print(response.text)
