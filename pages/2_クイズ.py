import streamlit as st
import random
import os
from PIL import Image

st.title("ひらがなクイズ")

romaji_to_hira = {
    "a":"あ","i":"い","u":"う","e":"え","o":"お",
    "ka":"か","ki":"き","ku":"く","ke":"け","ko":"こ",
    "sa":"さ","shi":"し","su":"す","se":"せ","so":"そ",
    "ta":"た","chi":"ち","tsu":"つ","te":"て","to":"と",
    "na":"な","ni":"に","nu":"ぬ","ne":"ね","no":"の",
    "ha":"は","hi":"ひ","fu":"ふ","he":"へ","ho":"ほ",
    "ma":"ま","mi":"み","mu":"む","me":"め","mo":"も",
    "ya":"や","yu":"ゆ","yo":"よ",
    "ra":"ら","ri":"り","ru":"る","re":"れ","ro":"ろ",
    "wa":"わ","wo":"を","n":"ん"
}

image_folder = "yubimoji_illust"

image_files = [
    f for f in os.listdir(image_folder)
    if f.endswith((".PNG", ".jpg", ".jpeg"))
]

if "quiz_image" not in st.session_state:
    st.session_state.quiz_image = random.choice(image_files)

image_file = st.session_state.quiz_image

st.image(
    os.path.join(image_folder, image_file),
    width=300
)

# ファイル名からローマ字取得
# 例: 6ka_A.PNG → ka

filename = os.path.splitext(image_file)[0]

romaji = ""

for c in filename:
    if c.isdigit():
        continue
    if c == "_":
        break
    romaji += c

answer = romaji_to_hira.get(
    romaji,
    romaji
)

# 初回のみ選択肢生成
if "choices" not in st.session_state:
    all_hira = list(romaji_to_hira.values())

    wrongs = random.sample(
        [x for x in all_hira if x != answer],
        3
    )

    choices = wrongs + [answer]

    random.shuffle(choices)

    st.session_state.choices = choices

selected = st.radio(
    "答えを選んでください",
    st.session_state.choices
)

if st.button("答え合わせ"):

    if selected == answer:
        st.success("正解！")
    else:
        st.error(
            f"不正解 正解は {answer}"
        )

if st.button("次の問題"):

    st.session_state.quiz_image = random.choice(
        image_files
    )

    if "choices" in st.session_state:
        del st.session_state["choices"]

    st.rerun()