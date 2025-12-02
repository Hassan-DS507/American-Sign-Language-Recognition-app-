# app.py
"""
ASL Recognition System - Full working version with:
 - improved camera opener (multiple backends)
 - file upload fallback (video/image) for remote deployments
 - demo mode when camera is unavailable
 - model loading with safe fallbacks
 - same UI & processing logic as original
"""

import os
import sys
import json
import time
import tempfile
import platform
from collections import deque

import numpy as np
import cv2
import mediapipe as mp
import streamlit as st

# ==========================================
# CONFIGURATION
# ==========================================

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable TF warnings

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.75

MODEL_FILES = {
    'minixgb': 'xgb_model.pkl',
    'xgboost': 'xgboost_asl.pkl',
    'bilstm': 'bilstm_model_2.keras'
}

LABEL_MAP_FILE = 'label_map_2.json'

# ==========================================
# SIMPLE STYLES
# ==========================================

st.set_page_config(
    page_title="ASL Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { text-align: center; color: #2563eb; font-size: 2.2rem; margin-bottom: 1rem; font-weight: bold; }
    .sub-title { text-align: center; color: #6b7280; margin-bottom: 1rem; }
    .prediction-box { background: #f8fafc; padding: 1.2rem; border-radius: 10px; border-left: 5px solid #2563eb; text-align: center; min-height: 180px; display: flex; flex-direction: column; justify-content: center; }
    .prediction-text { font-size: 2rem; font-weight: bold; color: #1e293b; margin: 0.5rem 0; }
    .confidence-bar { height: 18px; background: #e2e8f0; border-radius: 10px; margin: 0.6rem 0; overflow: hidden; }
    .confidence-fill { height: 100%; background: linear-gradient(90deg, #10b981, #059669); border-radius: 10px; transition: width 0.5s ease; }
    .camera-frame { border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); background: black; }
    .status-badge { padding: 5px 12px; border-radius: 20px; font-size: 0.9rem; font-weight: 600; display: inline-block; }
    .status-on { background: #dcfce7; color: #166534; }
    .status-off { background: #f3f4f6; color: #6b7280; }
    .status-error { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# INITIALIZE MEDIAPIPE
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("## Control Panel")

model_type = st.sidebar.selectbox(
    "AI Model",
    ["Mini-XGBoost (Fastest)", "XGBoost (Balanced)", "Bi-LSTM (Most Accurate)"]
)

conf_threshold = st.sidebar.slider(
    "Confidence Level",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05
)

show_skeleton = st.sidebar.checkbox("Show Skeleton", value=True)
show_fps = st.sidebar.checkbox("Show FPS", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Camera Settings")
use_camera = st.sidebar.checkbox("Enable Live Camera", value=True)

# Upload fallback for remote deployments
uploaded_file = st.sidebar.file_uploader("Upload video/image (fallback)", 
                                         type=["mp4", "mov", "avi", "mkv", "jpg", "png"])

if not use_camera:
    st.sidebar.info("Demo mode active - No camera required")

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_features(results):
    """Extract pose + left/right hand landmarks into a 198-length vector (floats)."""
    features = []
    # Pose 33 points * 2
    if results and results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y])
    else:
        features.extend([0.0] * 66)
    # Left hand 21 points * 3
    if results and results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    # Right hand 21 points * 3
    if results and results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    return np.array(features[:198], dtype=np.float32)

# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_model_cached(selected_model_label):
    """Load model and labels based on selection. Return (model, labels, is_deep_learning)"""
    try:
        if "Mini" in selected_model_label:
            # Mini XGBoost - joblib
            import joblib
            model = joblib.load(MODEL_FILES['minixgb'])
            labels = {0: 'banana', 1: 'jacket', 2: 'cry', 3: 'catch', 4: 'pop'}
            return model, labels, False
        elif "XGBoost" in selected_model_label and "Mini" not in selected_model_label:
            import joblib
            model = joblib.load(MODEL_FILES['xgboost'])
            is_dl = False
        else:
            # Bi-LSTM
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_FILES['bilstm'], compile=False)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            is_dl = True

        # load label map
        try:
            with open(LABEL_MAP_FILE, 'r') as f:
                label_data = json.load(f)
            if isinstance(label_data, dict):
                labels = {int(v): k for k, v in label_data.items()}
            else:
                labels = {i: label for i, label in enumerate(label_data)}
        except Exception:
            labels = {
                0: 'banana', 1: 'catch', 2: 'cool', 3: 'cry', 4: 'drown',
                5: 'envelope', 6: 'erase', 7: 'follow', 8: 'jacket', 9: 'pineapple',
                10: 'pop', 11: 'sandwich', 12: 'shave', 13: 'strawberry'
            }

        return model, labels, is_dl
    except Exception as e:
        # return None indicates failure
        st.error(f"Error loading model: {str(e)[:150]}")
        return None, None, False

model, labels, is_deep_learning = load_model_cached(model_type)

if model is None:
    st.error("""
    ### Model Loading Failed
    Ensure model files exist in the same folder as app.py:
    - Mini XGBoost: xgb_model.pkl
    - XGBoost: xgboost_asl.pkl and label_map_2.json
    - Bi-LSTM: bilstm_model_2.keras and label_map_2.json
    """)
    st.stop()

# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown('<h1 class="main-title">ASL Recognition System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time American Sign Language Translation</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Live Camera Feed")
    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("▶️ Start Camera", type="primary")
    with col_stop:
        stop_btn = st.button("⏹️ Stop Camera")

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        cam_status = st.empty()
    with status_col2:
        model_status = st.empty()

    camera_placeholder = st.empty()
    fps_display = st.empty()

with col2:
    st.markdown("### Recognition Results")
    result_placeholder = st.empty()
    buffer_display = st.empty()
    confidence_display = st.empty()

# ==========================================
# CAMERA UTILITIES (improved)
# ==========================================
def create_dummy_frame(message="Camera Not Available"):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    cv2.putText(frame, "ASL Recognition System", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 150, 255), 2)
    cv2.putText(frame, message, (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, "Demo Mode", (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
    cv2.circle(frame, (320, 320), 50, (100, 150, 255), 2)
    return frame

def try_open_camera():
    """Try to open camera with multiple indices and platform-specific backends."""
    indices = [0, 1, 2, -1]
    system = platform.system().lower()
    # Try raw indices
    for idx in indices:
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap, idx
                cap.release()
        except Exception:
            pass

    # Try platform-specific backends
    backends = []
    if system == "windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]
    elif system == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_QT]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER]

    for backend in backends:
        for idx in indices:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        return cap, idx
                    cap.release()
            except Exception:
                pass

    # final attempt default 0
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap, 0
            cap.release()
    except Exception:
        pass

    return None, -1

def save_uploaded_tempfile(uploaded):
    """Save uploaded file to a temporary path and return path."""
    suffix = ''
    if uploaded.type and 'video' in uploaded.type:
        suffix = os.path.splitext(uploaded.name)[1] or '.mp4'
    elif uploaded.name:
        suffix = os.path.splitext(uploaded.name)[1] or '.png'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    return tmp.name

# ==========================================
# SESSION STATE INIT
# ==========================================
if 'sequence' not in st.session_state:
    st.session_state.sequence = deque(maxlen=SEQUENCE_LENGTH)
if 'running' not in st.session_state:
    st.session_state.running = False
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = ("", 0.0)
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_time' not in st.session_state:
    st.session_state.last_time = time.time()
if 'uploaded_video_path' not in st.session_state:
    st.session_state.uploaded_video_path = None

# If user uploaded file - save it once
if uploaded_file and st.session_state.uploaded_video_path is None:
    try:
        path = save_uploaded_tempfile(uploaded_file)
        st.session_state.uploaded_video_path = path
        st.sidebar.success("Uploaded file saved for processing.")
    except Exception as e:
        st.sidebar.error(f"Failed to save uploaded file: {e}")

# ==========================================
# START / STOP CONTROLS
# ==========================================
if start_btn:
    st.session_state.running = True
    st.session_state.sequence.clear()
    st.session_state.predictions = []
    st.session_state.current_prediction = ("", 0.0)

if stop_btn:
    st.session_state.running = False
    # release camera if open
    try:
        if isinstance(st.session_state.camera, cv2.VideoCapture):
            st.session_state.camera.release()
    except Exception:
        pass
    st.session_state.camera = None

# ==========================================
# MAIN PROCESSING
# ==========================================
if st.session_state.running:
    cam_status.markdown('<span class="status-badge status-on">Camera: ON</span>', unsafe_allow_html=True)
    model_status.markdown('<span class="status-badge status-on">Model: READY</span>', unsafe_allow_html=True)

    # Initialize camera source if none
    if st.session_state.camera is None:
        if use_camera:
            cap, cam_idx = try_open_camera()
            if cap:
                st.session_state.camera = cap
                st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                st.success(f"Camera connected (Index: {cam_idx})")
            else:
                # Try uploaded video fallback
                if st.session_state.uploaded_video_path:
                    cap = cv2.VideoCapture(st.session_state.uploaded_video_path)
                    if cap.isOpened():
                        st.session_state.camera = cap
                        st.info("Using uploaded video as camera source.")
                    else:
                        st.warning("Could not open uploaded file. Switching to Demo Mode.")
                        st.session_state.camera = "demo"
                else:
                    st.warning("⚠️ Could not access camera. Switching to Demo Mode.")
                    st.session_state.camera = "demo"
        else:
            # camera disabled by user - use demo or uploaded video
            if st.session_state.uploaded_video_path:
                cap = cv2.VideoCapture(st.session_state.uploaded_video_path)
                if cap.isOpened():
                    st.session_state.camera = cap
                    st.info("Using uploaded video as camera source.")
                else:
                    st.session_state.camera = "demo"
            else:
                st.session_state.camera = "demo"
                st.info("Demo Mode Active - Using simulated camera feed")

    # Processing with MediaPipe
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    ) as holistic:

        frame_placeholder = camera_placeholder.empty()

        # Use broken loop guard so Streamlit UI remains responsive
        while st.session_state.running:
            # FPS calculations
            st.session_state.frame_count += 1
            current_time = time.time()
            elapsed = current_time - st.session_state.last_time
            if elapsed >= 1.0:
                st.session_state.fps = st.session_state.frame_count / elapsed
                st.session_state.frame_count = 0
                st.session_state.last_time = current_time

            # Acquire frame
            results = None
            if st.session_state.camera == "demo":
                frame = create_dummy_frame("Demo Mode - No Camera")
                results = None
            else:
                try:
                    ret, frame = st.session_state.camera.read()
                except Exception:
                    ret = False
                    frame = None

                if not ret or frame is None:
                    # If uploaded video ended, rewind or switch to demo
                    if st.session_state.uploaded_video_path and isinstance(st.session_state.camera, cv2.VideoCapture):
                        # try to rewind
                        try:
                            st.session_state.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = st.session_state.camera.read()
                            if not ret:
                                st.session_state.camera = "demo"
                                frame = create_dummy_frame("Uploaded Video Ended - Demo Mode")
                        except Exception:
                            st.session_state.camera = "demo"
                            frame = create_dummy_frame("Camera Error - Demo Mode")
                    else:
                        st.session_state.camera = "demo"
                        frame = create_dummy_frame("Camera Error - Demo Mode")
                else:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)

                    if show_skeleton and results:
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1),
                                mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1)
                            )
                        if results.left_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                            )
                        if results.right_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                            )

            # Feature extraction and buffering
            if results:
                features = extract_features(results)
                st.session_state.sequence.append(features)
            elif st.session_state.camera == "demo":
                if np.random.random() > 0.7:
                    demo_features = np.random.randn(198).astype(np.float32)
                    st.session_state.sequence.append(demo_features)

            buffer_len = len(st.session_state.sequence)
            buffer_percent = int((buffer_len / SEQUENCE_LENGTH) * 100)
            buffer_display.progress(min(buffer_len / SEQUENCE_LENGTH, 1.0), text=f"Buffer: {buffer_len}/{SEQUENCE_LENGTH}")

            # Prediction when buffer full
            current_word = ""
            current_conf = 0.0

            if buffer_len == SEQUENCE_LENGTH:
                try:
                    sequence_array = np.array(st.session_state.sequence)
                    if is_deep_learning:
                        input_seq = sequence_array.reshape(1, SEQUENCE_LENGTH, 198)
                        predictions = model.predict(input_seq, verbose=0)[0]
                        pred_idx = int(np.argmax(predictions))
                        current_conf = float(predictions[pred_idx])
                        current_word = labels.get(pred_idx, "Unknown")
                    else:
                        mean_features = np.mean(sequence_array, axis=0).reshape(1, -1)
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(mean_features)[0]
                            pred_idx = int(np.argmax(probabilities))
                            current_conf = float(probabilities[pred_idx])
                        else:
                            pred_idx = int(model.predict(mean_features)[0])
                            current_conf = 0.85
                        current_word = labels.get(pred_idx, "Unknown")

                    # store high-confidence results
                    if current_conf >= conf_threshold:
                        st.session_state.current_prediction = (current_word, current_conf)
                        st.session_state.predictions.append((current_word, current_conf))
                        if len(st.session_state.predictions) > 5:
                            st.session_state.predictions.pop(0)
                except Exception as e:
                    # graceful fallback in demo mode
                    if st.session_state.camera == "demo":
                        demo_words = list(labels.values())[:6] if labels else ['hello', 'thanks', 'yes', 'no', 'please']
                        current_word = np.random.choice(demo_words)
                        current_conf = np.random.uniform(0.7, 0.95)
                        st.session_state.current_prediction = (current_word, current_conf)

            # Update UI results
            word, conf = st.session_state.current_prediction

            if word:
                result_placeholder.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-text">{word.upper()}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {conf*100}%"></div>
                    </div>
                    <p style="color: #6b7280; font-size: 1rem;">
                        Confidence: {conf*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if st.session_state.camera != "demo" and 'frame' in locals() and frame is not None:
