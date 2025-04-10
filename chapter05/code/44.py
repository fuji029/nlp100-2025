from google import genai
import os

# 環境変数からAPIKEYを読み込み
client = genai.Client(api_key=os.environ["GEMINI_APIKEY"])

# 対話
chat = client.chats.create(model="gemini-1.5-flash")

contents = (
    "つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。"
    "東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。"
    "自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。"
)

response = chat.send_message(contents)

for message in chat.get_history():
    print(f"role - {message.role}", end=": ")
    print(message.parts[0].text.replace("\n", ""))
