import streamlit as st
import random
import base64
from pathlib import Path

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

image_folder = Path(__file__).resolve().parent.parent / "yubimoji_illust"

image_files = sorted(
    path for path in image_folder.iterdir()
    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
) if image_folder.is_dir() else []

if not image_files:
    st.error(f"クイズ画像が見つかりません。画像フォルダーを確認してください: {image_folder.name}")
    st.stop()

if st.session_state.get("quiz_image") not in image_files:
    st.session_state.quiz_image = random.choice(image_files)

image_file = st.session_state.quiz_image

image_bytes = base64.b64encode(image_file.read_bytes()).decode("ascii")
image_type = "image/jpeg" if image_file.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
st.markdown(
    f'<img src="data:{image_type};base64,{image_bytes}" ' +
    'style="width:300px;max-width:100%;height:auto;" alt="クイズ画像">',
    unsafe_allow_html=True,
)

# ファイル名からローマ字取得
# 例: 6ka_A.PNG → ka

filename = image_file.stem

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

