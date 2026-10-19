import pickle

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model


@st.cache_resource
def load_my_model():
    return load_model("all_new_5cross_2.h5", compile=False)


model = load_my_model()

with open("labelencoder_all_new.pkl", "rb") as f:
    label_encoder = pickle.load(f)

romaji_to_hira = {
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
    "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ",
    "sa": "さ", "shi": "し", "su": "す", "se": "せ", "so": "そ",
    "ta": "た", "chi": "ち", "tsu": "つ", "te": "て", "to": "と",
    "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
    "ha": "は", "hi": "ひ", "fu": "ふ", "he": "へ", "ho": "ほ",
    "ma": "ま", "mi": "み", "mu": "む", "me": "め", "mo": "も",
    "ya": "や", "yu": "ゆ", "yo": "よ",
    "ra": "ら", "ri": "り", "ru": "る", "re": "れ", "ro": "ろ",
    "wa": "わ", "wo": "を", "n": "ん",
}


def extract_features(hand_landmarks):
    landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
    relative = (landmarks - landmarks[0]).flatten()
    features = np.concatenate([relative, relative.copy()])
    return features.reshape(1, -1)


def draw_japanese_text(frame, text, font):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text((20, 20), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


st.title("字幕機能")
st.write("手の形をカメラで撮影すると、認識結果が表示されます。")

if "subtitle_text" not in st.session_state:
    st.session_state.subtitle_text = ""
if "current_hira" not in st.session_state:
    st.session_state.current_hira = ""

font = ImageFont.truetype(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 40
)

photo = st.camera_input("手の形を撮影")

if photo is not None:
    rgb = np.array(Image.open(photo).convert("RGB"))
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        prediction = model.predict(extract_features(hand_landmarks), verbose=0).squeeze()
        idx = int(np.argmax(prediction))
        romaji = label_encoder.inverse_transform([idx])[0]
        confidence = float(np.max(prediction))
        hira = romaji_to_hira.get(romaji, romaji)
        st.session_state.current_hira = hira

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        frame = draw_japanese_text(
            frame, f"認識結果：{hira} ({confidence:.1%})", font
        )
        st.image(frame, channels="BGR", caption="認識結果")
    else:
        st.session_state.current_hira = ""
        st.image(rgb, caption="手を認識できませんでした。手を画面に入れて撮り直してください。")

if st.session_state.current_hira:
    st.info(f"現在の文字：{st.session_state.current_hira}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("文字確定") and st.session_state.current_hira:
        st.session_state.subtitle_text += st.session_state.current_hira
with col2:
    if st.button("1文字削除"):
        st.session_state.subtitle_text = st.session_state.subtitle_text[:-1]
with col3:
    if st.button("字幕クリア"):
        st.session_state.subtitle_text = ""
with col4:
    if st.button("改行"):
        st.session_state.subtitle_text += "\n"

st.subheader("字幕")
st.text(st.session_state.subtitle_text)
