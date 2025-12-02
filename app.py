import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
import joblib
import json
import av
import os
from collections import deque

# ==========================================
# 1. إعدادات الصفحة والتصميم
# ==========================================
st.set_page_config(page_title="ASL AI Recognition", page_icon="🤟", layout="wide")

st.markdown("""
<style>
    .main-title {font-size: 2.5rem; color: #4F8BF9; text-align: center; font-weight: 800;}
    .status-box {padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;}
    .success {background-color: #d4edda; color: #155724;}
    .warning {background-color: #fff3cd; color: #856404;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. تحميل الموارد (موديلات وملفات)
# ==========================================
# مسارات الملفات
MODEL_FILES = {
    'bilstm': 'bilstm_model_2.keras',
    'xgboost': 'xgboost_asl.pkl',
    'minixgb': 'xgb_model.pkl'
}
LABEL_FILE = 'label_map_2.json'

@st.cache_resource
def load_resources():
    # تحميل TensorFlow فقط عند الحاجة لتوفير الذاكرة
    try:
        import tensorflow as tf
    except ImportError:
        tf = None

    models = {}
    
    # 1. تحميل XGBoost (سريع وخفيف)
    if os.path.exists(MODEL_FILES['xgboost']):
        models['xgboost'] = joblib.load(MODEL_FILES['xgboost'])
    
    if os.path.exists(MODEL_FILES['minixgb']):
        models['minixgb'] = joblib.load(MODEL_FILES['minixgb'])

    # 2. تحميل Keras (يحتاج TensorFlow)
    if tf and os.path.exists(MODEL_FILES['bilstm']):
        try:
            models['bilstm'] = tf.keras.models.load_model(MODEL_FILES['bilstm'], compile=False)
        except:
            pass

    # 3. تحميل الأسماء
    labels = {}
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, 'r') as f:
            data = json.load(f)
            # ضبط التنسيق {0: "word"}
            if isinstance(data, dict):
                first_val = list(data.values())[0]
                if isinstance(first_val, int): # الشكل {word: 0}
                    labels = {v: k for k, v in data.items()}
                else: # الشكل {0: word}
                    labels = {int(k): v for k, v in data.items()}
    else:
        # قيم افتراضية للطوارئ
        labels = {i: f"Sign {i}" for i in range(20)}

    return models, labels, tf

# تحميل الموارد مرة واحدة
models_dict, label_map, tf_module = load_resources()

# ==========================================
# 3. دوال المعالجة (Feature Extraction)
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_features_v3(results):
    feat = np.zeros(198, dtype=np.float32)
    
    # Pose
    torso_center = np.array([0.5, 0.5], dtype=np.float32)
    torso_scale = 1.0
    
    if results.pose_landmarks:
        # حساب مركز وحجم الجسم
        landmarks = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark], dtype=np.float32)
        left_sh, right_sh = landmarks[11], landmarks[12]
        left_hip, right_hip = landmarks[23], landmarks[24]
        
        shoulder_center = (left_sh + right_sh) / 2
        hip_center = (left_hip + right_hip) / 2
        torso_center = (shoulder_center + hip_center) / 2
        torso_scale = max(np.linalg.norm(left_sh - right_sh), np.linalg.norm(left_hip - right_hip), 1e-6)
        
        # تطبيع الجسم
        pose_norm = (landmarks - torso_center) / torso_scale
        feat[0:66] = pose_norm.flatten()

    # اليدين (دالة مساعدة للتطبيع)
    def process_hand(hand_landmarks, start_idx):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        # تطبيع اليد بالنسبة للمعصم
        wrist = pts[0]
        scale = np.linalg.norm(pts[9] - wrist) # المسافة للأصبع الوسطى
        scale = scale if scale > 1e-6 else 1.0
        hand_norm = (pts - wrist) / scale
        feat[start_idx : start_idx+63] = hand_norm[:, :3].flatten()
        # مكان المعصم بالنسبة للجسم
        wrist_rel = wrist.copy()
        wrist_rel[:2] = (wrist[:2] - torso_center) / torso_scale
        wrist_rel[2] /= torso_scale
        feat[start_idx+63 : start_idx+66] = wrist_rel

    if results.left_hand_landmarks:
        process_hand(results.left_hand_landmarks, 66)
        
    if results.right_hand_landmarks:
        process_hand(results.right_hand_landmarks, 132)
        
    return feat

def prepare_for_xgboost(buffer):
    data = np.array(buffer, dtype=np.float32)
    return np.concatenate([
        np.mean(data, axis=0),
        np.std(data, axis=0),
        np.max(data, axis=0),
        np.mean(np.diff(data, axis=0), axis=0)
    ]).reshape(1, -1)

# ==========================================
# 4. معالج الفيديو (Video Transformer)
# ==========================================
# هذا الكلاس هو المسؤول عن معالجة الفيديو فريم-بفريم داخل WebRTC
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence = deque(maxlen=50)
        self.prediction = "..."
        self.confidence = 0.0
        
        # قراءة الإعدادات من الـ Session State
        self.model_key = st.session_state.get('model_type', 'xgboost')
        self.threshold = st.session_state.get('conf_threshold', 0.75)
        self.draw = st.session_state.get('draw_skeleton', True)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. معالجة MediaPipe
        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        img.flags.writeable = True
        
        # 2. الرسم
        if self.draw:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # 3. استخراج البيانات والتوقع
        keypoints = extract_features_v3(results)
        self.sequence.append(keypoints)
        
        if len(self.sequence) == 50:
            try:
                model = models_dict.get(self.model_key)
                if model:
                    # تجهيز الداتا حسب نوع الموديل
                    if self.model_key == 'bilstm' and tf_module:
                        input_data = np.expand_dims(self.sequence, axis=0).astype(np.float32)
                        res = model.predict(input_data, verbose=0)[0]
                    else: # XGBoost variants
                        input_data = prepare_for_xgboost(self.sequence)
                        res = model.predict_proba(input_data)[0]
                    
                    idx = np.argmax(res)
                    conf = res[idx]
                    
                    if conf > self.threshold:
                        self.prediction = label_map.get(idx, "Unknown")
                        self.confidence = conf
                    else:
                        self.prediction = "..."
                        self.confidence = 0.0
            except Exception as e:
                print(f"Error: {e}")

        # 4. الكتابة على الفيديو
        cv2.rectangle(img, (0,0), (640, 60), (0,0,0), -1)
        cv2.putText(img, f"{self.prediction.upper()} ({int(self.confidence*100)}%)", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img

# ==========================================
# 5. واجهة المستخدم (UI)
# ==========================================
st.markdown("<div class='main-title'>ASL Recognition System</div>", unsafe_allow_html=True)

# الشريط الجانبي للإعدادات
with st.sidebar:
    st.header("⚙️ Settings")
    
    # اختيار الموديل وحفظه في Session State
    model_choice = st.selectbox("Choose Model", 
                                ["XGBoost (Fast)", "Bi-LSTM (Accurate)", "Mini-XGBoost"], 
                                key="model_select")
    
    if "Bi-LSTM" in model_choice:
        st.session_state['model_type'] = 'bilstm'
    elif "Mini" in model_choice:
        st.session_state['model_type'] = 'minixgb'
    else:
        st.session_state['model_type'] = 'xgboost'

    # إعدادات أخرى
    st.session_state['conf_threshold'] = st.slider("Confidence Threshold", 0.5, 1.0, 0.75)
    st.session_state['draw_skeleton'] = st.checkbox("Draw Skeleton", value=True)
    
    st.markdown("---")
    st.info("**Instructions:**\n1. Select model.\n2. Click 'Start' below.\n3. Allow camera access.\n4. Perform gestures.")

# منطقة الكاميرا (WebRTC)
col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.markdown("### 📷 Live Camera")
    
    # إعدادات WebRTC للسيرفر (ضرورية للكلاود)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="asl-stream",
        mode=webrtc_streamer.WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=ASLTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("### 📊 Status")
    
    # عرض حالة الموديلات
    if models_dict.get(st.session_state['model_type']):
        st.markdown(f"<div class='status-box success'>Model Loaded: {model_choice}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='status-box warning'>Model Not Found!</div>", unsafe_allow_html=True)
        st.caption("Please verify model files exist in the directory.")

    st.markdown("---")
    st.markdown("#### Loaded Vocabulary")
    st.write(list(label_map.values())[:10]) # عرض أول 10 كلمات فقط