import pickle
import threading

import av
import cv2
import mediapipe as mp
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from tensorflow.keras.models import load_model


@st.cache_resource
def load_my_model():
    return load_model("all_new_5cross_2.h5", compile=False)


model = load_my_model()


@st.cache_data(ttl=300)
def get_ice_servers(turn_key_id, turn_api_token):
    """Fetch expiring Cloudflare TURN credentials if configured."""
    if not turn_key_id or not turn_api_token:
        return [{"urls": ["stun:stun.l.google.com:19302"]}], False

    response = requests.post(
        "https://rtc.live.cloudflare.com/v1/turn/keys/"
        f"{turn_key_id}/credentials/generate-ice-servers",
        headers={"Authorization": f"Bearer {turn_api_token}"},
        json={"ttl": 3600},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["iceServers"], True


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


class SignProcessor(VideoProcessorBase):
    def __init__(self, shared_result, result_lock):
        self.shared_result = shared_result
        self.result_lock = result_lock
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 40
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hira = ""
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            prediction = model.predict(
                extract_features(hand_landmarks), verbose=0
            ).squeeze()
            idx = int(np.argmax(prediction))
            romaji = label_encoder.inverse_transform([idx])[0]
            confidence = float(np.max(prediction))
            hira = romaji_to_hira.get(romaji, romaji)
            self.mp_draw.draw_landmarks(
                image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )
            overlay = f"認識中：{hira} ({confidence:.1%})"
        else:
            overlay = "認識中：－"

        with self.result_lock:
            self.shared_result["hira"] = hira
            self.shared_result["confidence"] = confidence

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil_image).text(
            (20, 20), overlay, font=self.font, fill=(255, 255, 255)
        )
        output = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(output, format="bgr24")


st.title("字幕機能")
st.write("カメラに手を映すと、骨格と認識結果がリアルタイムで表示されます。")

if "subtitle_text" not in st.session_state:
    st.session_state.subtitle_text = ""
if "current_hira" not in st.session_state:
    st.session_state.current_hira = ""
if "sign_result" not in st.session_state:
    st.session_state.sign_result = {"hira": "", "confidence": 0.0}
if "sign_result_lock" not in st.session_state:
    st.session_state.sign_result_lock = threading.Lock()

# Capture state on the Streamlit script thread. The WebRTC processor runs
# on a separate thread and must not access st.session_state directly.
shared_result = st.session_state.sign_result
result_lock = st.session_state.sign_result_lock

try:
    turn_key_id = st.secrets.get("CLOUDFLARE_TURN_KEY_ID", "")
    turn_api_token = st.secrets.get("CLOUDFLARE_TURN_API_TOKEN", "")
except Exception:
    turn_key_id = ""
    turn_api_token = ""

try:
    ice_servers, turn_enabled = get_ice_servers(turn_key_id, turn_api_token)
except requests.RequestException:
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    turn_enabled = False
    st.warning("TURNサーバーに接続できません。STUNでカメラ接続を試します。")

if not turn_enabled:
    st.info("接続できない場合は、Cloudflare TURNキーをStreamlit Secretsに設定してください。")

webrtc_ctx = webrtc_streamer(
    key="sign-recognition-camera",
    video_processor_factory=lambda: SignProcessor(shared_result, result_lock),
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": ice_servers},
)


@st.fragment(run_every="300ms")
def show_current_recognition():
    with result_lock:
        result = dict(shared_result)
    st.session_state.current_hira = result["hira"]
    if result["hira"]:
        st.caption(f"現在の認識文字：{result['hira']}（{result['confidence']:.1%}）")
    elif webrtc_ctx.state.playing:
        st.caption("手をカメラに映してください")
    else:
        st.caption("カメラの開始ボタンを押してください")


show_current_recognition()

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
