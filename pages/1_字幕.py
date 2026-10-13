import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

@st.cache_resource
def load_my_model():
    return load_model(
        "all_new_5cross_2.h5",
        compile=False
    )

model = load_my_model()

with open("labelencoder_all_new.pkl", "rb") as f:
    label_encoder = pickle.load(f)
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

def extract_features(hand_landmarks):

    landmarks = np.array(
        [[lm.x, lm.y] for lm in hand_landmarks.landmark]
    )

    base = landmarks[0]

    relative = (landmarks - base).flatten()

    features = np.concatenate(
        [relative, relative.copy()]
    )

    return features.reshape(1, -1)

def draw_japanese_text(frame, text):
    # OpenCV(BGR) → PIL(RGB)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(image)

    draw.text(
        (20, 20),
        text,
        font=font,
        fill=(255, 255, 255)
    )

    # PIL(RGB) → OpenCV(BGR)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    return frame



st.title("字幕機能")

font = ImageFont.truetype(
    "C:/Windows/Fonts/meiryo.ttc",
    40
)

if "subtitle_text" not in st.session_state:
    st.session_state.subtitle_text = ""

if "current_hira" not in st.session_state:
    st.session_state.current_hira = ""

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

run = st.checkbox("カメラ起動")

frame_window = st.empty()

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("文字確定"):
        if st.session_state.current_hira:
            st.session_state.subtitle_text += (
                st.session_state.current_hira
            )

with col2:
    if st.button("1文字削除"):
        st.session_state.subtitle_text = (
            st.session_state.subtitle_text[:-1]
        )

with col3:
    if st.button("字幕クリア"):
        st.session_state.subtitle_text = ""

with col4:
    if st.button("改行"):
        st.session_state.subtitle_text += "\n"

st.subheader("字幕")

subtitle_area = st.empty()

subtitle_area.text(
    st.session_state.subtitle_text
)



if run:
    cap = cv2.VideoCapture(0)

    

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("カメラ取得失敗")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                features = extract_features(hand_landmarks)

                prediction = model.predict(
                   features,
                   verbose=0
               ).squeeze()

                idx = np.argmax(prediction)

                romaji = label_encoder.inverse_transform([idx])[0]

                confidence = np.max(prediction)

                hira = romaji_to_hira.get(
                    romaji,
                    romaji
                )
                
                st.session_state.current_hira = hira

                frame = draw_japanese_text(
                    frame,
                    f"認識中：{hira} ({confidence:.1%})"
                )

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                   mp_hands.HAND_CONNECTIONS
                )
        else:
            frame = draw_japanese_text(
            frame,
            "認識中：－"
        )
        frame_window.image(
            frame,
            channels="BGR"
         )
    cap.release()

