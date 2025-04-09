from google import genai
import os

# 環境変数からAPIKEYを読み込み
client = genai.Client(api_key=os.environ["GEMINI_APIKEY"])

contents = (
    "9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。\n\n"
    "ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。\n"
    "イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。\n"
    "ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。\n"
)

response = client.models.generate_content(
    model="gemini-1.5-flash", contents=contents
)

print(response.text)
