from google import genai
import os

# 環境変数からAPIKEYを読み込み
client = genai.Client(api_key=os.environ["GEMINI_APIKEY"])

contents = (
    "あなたは日本の歌人です。次の川柳の面白さを10段階で評価して。出力は必ず1から10の数字のみで答えること。\n"
)

haiku = [
    "雀の子　つぎつぎ飛び立つ　春の空",
    "氷解け　川の流れ出す　春の音",
    "菜の花や　風そよぐ里　春の息",
    "桜咲く　頬を染めて　春の顔",
    "梅の花　香りに誘われ　春の歩",
    "新芽吹き　木々生き生きと　春の力",
    "雛人形　笑顔見守る　春の祝",
    "春の雨　土を潤して　芽をだす",
    "花冷えや　薄着寒し　春の罠",
    "夕焼け空　春の終わり　茜色",
]

for i in range(len(haiku)):
    response = client.models.generate_content(
        model="gemini-1.5-flash", contents=contents+haiku[i]
    )

    print(response.text.replace("\n", ""))
