import os
import sys
import json
import time
import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
from collections import deque

# ==========================================
# CONFIGURATION
# ==========================================

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constants
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.75

# Model files
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
    .main-title {
        text-align: center;
        color: #2563eb;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-title {
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2563eb;
        text-align: center;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .prediction-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e293b;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 20px;
        background: #e2e8f0;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #059669);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .camera-frame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background: black;
    }
    .status-badge {
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-on {
        background: #dcfce7;
        color: #166534;
    }
    .status-off {
        background: #f3f4f6;
        color: #6b7280;
    }
    .status-error {
        background: #fee2e2;
        color: #991b1b;
    }
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

# Model Selection
model_type = st.sidebar.selectbox(
    "AI Model",
    ["Mini-XGBoost (Fastest)", "XGBoost (Balanced)", "Bi-LSTM (Most Accurate)"]
)

# Confidence
conf_threshold = st.sidebar.slider(
    "Confidence Level",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05
)

# Display Options
show_skeleton = st.sidebar.checkbox("Show Skeleton", value=True)
show_fps = st.sidebar.checkbox("Show FPS", value=True)

# Camera Selection
st.sidebar.markdown("---")
st.sidebar.markdown("### Camera Settings")
use_camera = st.sidebar.checkbox("Enable Live Camera", value=True)

# If camera doesn't work, show demo mode
if not use_camera:
    st.sidebar.info("Demo mode active - No camera required")

# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(results):
    """Extract hand and pose landmarks"""
    features = []
    
    # Pose (33 points, 2 coordinates each = 66)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y])
    else:
        features.extend([0.0] * 66)
    
    # Left hand (21 points, 3 coordinates each = 63)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # Right hand (21 points, 3 coordinates each = 63)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # Ensure exactly 198 features
    return np.array(features[:198], dtype=np.float32)

# ==========================================
# MODEL LOADING
# ==========================================

@st.cache_resource
def load_model():
    """Load the selected model"""
    try:
        # Select model file
        if "Mini" in model_type:
            model_file = MODEL_FILES['minixgb']
            import joblib
            model = joblib.load(model_file)
            # Hardcoded labels for mini model
            labels = {0: 'banana', 1: 'jacket', 2: 'cry', 3: 'catch', 4: 'pop'}
            return model, labels, False
            
        elif "XGBoost" in model_type and "Mini" not in model_type:
            model_file = MODEL_FILES['xgboost']
            import joblib
            model = joblib.load(model_file)
            is_dl = False
            
        else:  # Bi-LSTM
            model_file = MODEL_FILES['bilstm']
            import tensorflow as tf
            model = tf.keras.models.load_model(model_file, compile=False)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            is_dl = True
        
        # Load labels from JSON file
        try:
            with open(LABEL_MAP_FILE, 'r') as f:
                label_data = json.load(f)
            
            if isinstance(label_data, dict):
                labels = {int(v): k for k, v in label_data.items()}
            else:
                labels = {i: label for i, label in enumerate(label_data)}
                
        except:
            # Default fallback labels
            labels = {
                0: 'banana', 1: 'catch', 2: 'cool', 3: 'cry', 4: 'drown',
                5: 'envelope', 6: 'erase', 7: 'follow', 8: 'jacket', 9: 'pineapple',
                10: 'pop', 11: 'sandwich', 12: 'shave', 13: 'strawberry'
            }
        
        return model, labels, is_dl
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)[:100]}")
        return None, None, False

# Load model
model, labels, is_deep_learning = load_model()

if model is None:
    st.error("""
    ### Model Loading Failed
    
    Please ensure these files are in the same folder:
    
    1. **For Mini-XGBoost:** `xgb_model.pkl`
    2. **For XGBoost:** `xgboost_asl.pkl` and `label_map_2.json`
    3. **For Bi-LSTM:** `bilstm_model_2.keras` and `label_map_2.json`
    
    **Current selection:** {model_type}
    """)
    st.stop()

# ==========================================
# MAIN INTERFACE
# ==========================================

# Header
st.markdown('<h1 class="main-title">ASL Recognition System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time American Sign Language Translation</p>', unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Live Camera Feed")
    
    # Camera control buttons
    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("▶️ Start Camera", type="primary", use_container_width=True)
    with col_stop:
        stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)
    
    # Camera status
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        cam_status = st.empty()
    with status_col2:
        model_status = st.empty()
    
    # Camera feed placeholder
    camera_placeholder = st.empty()
    
    # FPS display
    fps_display = st.empty()

with col2:
    st.markdown("### Recognition Results")
    
    # Results placeholder
    result_placeholder = st.empty()
    
    # Buffer display
    buffer_display = st.empty()
    
    # Confidence display
    confidence_display = st.empty()

# ==========================================
# CAMERA UTILITIES
# ==========================================

def create_dummy_frame(message="Camera Not Available"):
    """Create a dummy frame when camera is not available"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # Dark gray
    
    # Add text
    cv2.putText(frame, "ASL Recognition System", (80, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 150, 255), 2)
    cv2.putText(frame, message, (120, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, "Using Demo Mode", (180, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    
    # Draw a simple hand icon
    center_x, center_y = 320, 320
    cv2.circle(frame, (center_x, center_y), 50, (100, 150, 255), 2)
    
    return frame

def try_open_camera():
    """Try to open camera with multiple methods"""
    # Try different camera indices
    for camera_index in [0, 1, 2, -1]:
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                # Try to read a frame to confirm
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap, camera_index
                cap.release()
        except:
            continue
    
    # Try with different backend
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap, 0
    except:
        pass
    
    return None, -1

# ==========================================
# INITIALIZE SESSION STATE
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
    st.session_state.current_prediction = ("", 0)
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_time' not in st.session_state:
    st.session_state.last_time = time.time()

# ==========================================
# START/STOP CAMERA
# ==========================================

if start_btn:
    st.session_state.running = True
    st.session_state.sequence.clear()
    st.session_state.predictions = []
    st.session_state.current_prediction = ("", 0)

if stop_btn:
    st.session_state.running = False
    if st.session_state.camera:
        st.session_state.camera.release()
        st.session_state.camera = None

# ==========================================
# MAIN PROCESSING LOOP
# ==========================================

if st.session_state.running:
    # Update status
    cam_status.markdown('<span class="status-badge status-on">Camera: ON</span>', unsafe_allow_html=True)
    model_status.markdown('<span class="status-badge status-on">Model: READY</span>', unsafe_allow_html=True)
    
    # Initialize or get camera
    if st.session_state.camera is None:
        if use_camera:
            cap, cam_idx = try_open_camera()
            if cap:
                st.session_state.camera = cap
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                st.success(f"Camera connected (Index: {cam_idx})")
            else:
                st.warning("⚠️ Could not access camera. Switching to Demo Mode.")
                st.session_state.camera = "demo"
        else:
            st.session_state.camera = "demo"
            st.info("Demo Mode Active - Using simulated camera feed")
    
    # Process frames
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0  # Reduced for speed
    ) as holistic:
        
        frame_placeholder = camera_placeholder.empty()
        
        while st.session_state.running:
            # Calculate FPS
            st.session_state.frame_count += 1
            current_time = time.time()
            elapsed = current_time - st.session_state.last_time
            
            if elapsed >= 1.0:
                st.session_state.fps = st.session_state.frame_count / elapsed
                st.session_state.frame_count = 0
                st.session_state.last_time = current_time
            
            # Get frame
            if st.session_state.camera == "demo":
                # Demo mode - use dummy frame
                frame = create_dummy_frame("Demo Mode - No Camera")
                results = None
            else:
                # Live camera
                ret, frame = st.session_state.camera.read()
                if not ret:
                    # Camera error, switch to demo
                    st.session_state.camera = "demo"
                    frame = create_dummy_frame("Camera Error - Demo Mode")
                    results = None
                else:
                    # Process frame
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    
                    # Draw landmarks
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
            
            # Extract features
            if results:
                features = extract_features(results)
                st.session_state.sequence.append(features)
            elif st.session_state.camera == "demo":
                # Demo mode - add random features occasionally
                if np.random.random() > 0.7:  # 30% chance
                    demo_features = np.random.randn(198).astype(np.float32)
                    st.session_state.sequence.append(demo_features)
            
            # Update buffer display
            buffer_len = len(st.session_state.sequence)
            buffer_percent = int((buffer_len / SEQUENCE_LENGTH) * 100)
            
            buffer_display.progress(buffer_percent / 100, 
                                   text=f"Buffer: {buffer_len}/{SEQUENCE_LENGTH}")
            
            # Make prediction when buffer is full
            current_word = ""
            current_conf = 0
            
            if buffer_len == SEQUENCE_LENGTH:
                try:
                    sequence_array = np.array(st.session_state.sequence)
                    
                    if is_deep_learning:
                        # Bi-LSTM prediction
                        input_seq = sequence_array.reshape(1, SEQUENCE_LENGTH, 198)
                        predictions = model.predict(input_seq, verbose=0)[0]
                        pred_idx = np.argmax(predictions)
                        current_conf = float(predictions[pred_idx])
                        current_word = labels.get(pred_idx, "Unknown")
                    else:
                        # XGBoost prediction - use mean features
                        mean_features = np.mean(sequence_array, axis=0).reshape(1, -1)
                        
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(mean_features)[0]
                            pred_idx = np.argmax(probabilities)
                            current_conf = float(probabilities[pred_idx])
                        else:
                            pred_idx = model.predict(mean_features)[0]
                            current_conf = 0.85  # Default confidence for demo
                        
                        current_word = labels.get(int(pred_idx), "Unknown")
                    
                    # Store if confidence is high enough
                    if current_conf > conf_threshold:
                        st.session_state.current_prediction = (current_word, current_conf)
                        st.session_state.predictions.append((current_word, current_conf))
                        if len(st.session_state.predictions) > 5:
                            st.session_state.predictions.pop(0)
                            
                except Exception as e:
                    # For demo mode, generate random predictions
                    if st.session_state.camera == "demo":
                        demo_words = list(labels.values())[:5] if labels else ['hello', 'thanks', 'yes', 'no', 'please']
                        current_word = np.random.choice(demo_words)
                        current_conf = np.random.uniform(0.7, 0.95)
                        st.session_state.current_prediction = (current_word, current_conf)
            
            # Update displays
            word, conf = st.session_state.current_prediction
            
            if word:
                # Show prediction
                result_placeholder.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-text">{word.upper()}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {conf*100}%"></div>
                    </div>
                    <p style="color: #6b7280; font-size: 1.2rem;">
                        Confidence: {conf*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add to frame
                if st.session_state.camera != "demo":
                    cv2.putText(frame, f"{word.upper()} ({conf*100:.0f}%)", 
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Show waiting state
                result_placeholder.markdown(f"""
                <div class="prediction-box">
                    <h3>Analyzing Gesture</h3>
                    <p>Collecting data...</p>
                    <p>{buffer_len}/{SEQUENCE_LENGTH} frames</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {buffer_percent}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add FPS to frame
            if show_fps and st.session_state.camera != "demo":
                cv2.putText(frame, f"FPS: {int(st.session_state.fps)}", 
                           (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Update FPS display
            fps_display.metric("FPS", f"{int(st.session_state.fps) if st.session_state.fps > 0 else '--'}")
            
            # Display frame
            frame_placeholder.image(frame, channels="BGR")
            
            # Small delay to prevent freezing
            time.sleep(0.03)
    
    # Cleanup
    if st.session_state.camera and st.session_state.camera != "demo":
        st.session_state.camera.release()
    
    # Update status
    cam_status.markdown('<span class="status-badge status-off">Camera: OFF</span>', unsafe_allow_html=True)
    model_status.markdown('<span class="status-badge status-off">Model: IDLE</span>', unsafe_allow_html=True)
    
else:
    # Not running state
    cam_status.markdown('<span class="status-badge status-off">Camera: OFF</span>', unsafe_allow_html=True)
    model_status.markdown('<span class="status-badge status-off">Model: IDLE</span>', unsafe_allow_html=True)
    
    # Show default camera view
    default_frame = create_dummy_frame("Click 'Start Camera' to begin")
    camera_placeholder.image(default_frame, channels="BGR")
    
    # Show default result
    result_placeholder.markdown("""
    <div class="prediction-box">
        <h3>Ready for Recognition</h3>
        <p>Click "Start Camera" to begin</p>
        <p>Or disable camera for demo mode</p>
    </div>
    """, unsafe_allow_html=True)
    
    fps_display.metric("FPS", "--")
    buffer_display.progress(0, text="Buffer: 0/30")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Quick Guide
    1. Select AI model
    2. Click Start Camera
    3. Perform clear signs
    4. View results
    """)

with col2:
    st.markdown("""
    ### Camera Tips
    - Allow camera access
    - Good lighting
    - Steady hands
    - Clear background
    """)

with col3:
    st.markdown("""
    ### System Info
    - Frame buffer: 30
    - Real-time processing
    - Multiple AI models
    - Confidence scoring
    """)

st.markdown("---")
st.caption(f"ASL Recognition System | Model: {model_type}")

# ==========================================
# TROUBLESHOOTING
# ==========================================

with st.expander("Camera Help"):
    st.markdown("""
    ### Common Camera Issues
    
    **1. Camera not working in browser:**
    - Click the lock/camera icon in address bar
    - Allow camera permissions
    - Refresh the page
    
    **2. No camera detected:**
    - Ensure camera is connected
    - Try a different USB port
    - Test camera in another app
    
    **3. Camera works but freezes:**
    - Close other apps using camera
    - Reduce browser tabs
    - Restart browser
    
    **4. Mobile/Streamlit Cloud:**
    - Use Demo Mode (disable camera)
    - System works without camera
    - AI models still function
    
    **Demo Mode:**
    - No camera required
    - Simulated predictions
    - Perfect for testing
    - Works everywhere
    """)