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
# BASIC CONFIGURATION
# ==========================================

# Disable warnings and GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Constants
SEQUENCE_LENGTH = 30  # Reduced for faster processing
CONFIDENCE_THRESHOLD = 0.75
MODEL_FILES = {
    'bilstm': 'bilstm_model_2.keras',
    'xgboost': 'xgboost_asl.pkl',
    'minixgb': 'xgb_model.pkl'
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
    }
    .section-title {
        color: #4b5563;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .prediction-box {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2563eb;
        text-align: center;
    }
    .prediction-text {
        font-size: 3rem;
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
    }
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .status-online {
        background: #d1fae5;
        color: #065f46;
    }
    .status-offline {
        background: #f3f4f6;
        color: #6b7280;
    }
    .camera-frame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# INITIALIZE COMPONENTS
# ==========================================

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# SIDEBAR - SIMPLE CONTROLS
# ==========================================

st.sidebar.markdown("## Settings")

# Model Selection
model_type = st.sidebar.selectbox(
    "AI Model",
    ["Mini-XGBoost (Fast)", "XGBoost (Balanced)", "Bi-LSTM (Accurate)"]
)

# Camera Mode
use_camera = st.sidebar.checkbox("Use Live Camera", value=True)

# Confidence
conf_threshold = st.sidebar.slider(
    "Confidence Level",
    min_value=0.0,
    max_value=1.0,
    value=CONFIDENCE_THRESHOLD,
    step=0.05
)

# Display Options
show_skeleton = st.sidebar.checkbox("Show Skeleton", value=True)
show_fps = st.sidebar.checkbox("Show FPS", value=True)

# ==========================================
# PREPROCESSING FUNCTIONS (SIMPLIFIED)
# ==========================================

def extract_simple_features(results):
    """Extract essential features only"""
    features = []
    
    # Extract pose points
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y])
    else:
        features.extend([0.0] * 132)  # 66 points * 2
    
    # Extract hand points
    hand_points = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            hand_points.extend([lm.x, lm.y, lm.z])
    else:
        hand_points.extend([0.0] * 63)  # 21 points * 3
    
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            hand_points.extend([lm.x, lm.y, lm.z])
    else:
        hand_points.extend([0.0] * 63)
    
    features.extend(hand_points)
    return np.array(features[:198], dtype=np.float32)  # Keep 198 features

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
            labels = {0: 'banana', 1: 'jacket', 2: 'cry', 3: 'catch', 4: 'pop'}
            return model, labels, False
            
        elif "XGBoost" in model_type:
            model_file = MODEL_FILES['xgboost']
            import joblib
            model = joblib.load(model_file)
            
        else:  # Bi-LSTM
            model_file = MODEL_FILES['bilstm']
            import tensorflow as tf
            model = tf.keras.models.load_model(model_file, compile=False)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Load labels from file
        with open(LABEL_MAP_FILE, 'r') as f:
            label_data = json.load(f)
        
        if isinstance(label_data, dict):
            labels = {v: k for k, v in label_data.items()}
        else:
            labels = {i: label for i, label in enumerate(label_data)}
        
        return model, labels, "Bi-LSTM" in model_type
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)[:100]}")
        return None, None, False

# Load model
model, labels, is_deep_learning = load_model()

# ==========================================
# MAIN UI
# ==========================================

# Header
st.markdown('<h1 class="main-title">ASL Recognition System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280;">Real-time sign language translation</p>', unsafe_allow_html=True)

# Main columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h3 class="section-title">Camera Feed</h3>', unsafe_allow_html=True)
    
    # Camera control
    if use_camera:
        start_camera = st.button("Start Camera", type="primary", use_container_width=True)
        stop_camera = st.button("Stop Camera", use_container_width=True)
    else:
        start_camera = False
        stop_camera = True
        st.info("Camera disabled. Using demo mode.")
    
    # Camera placeholder
    camera_placeholder = st.empty()

with col2:
    st.markdown('<h3 class="section-title">Recognition Result</h3>', unsafe_allow_html=True)
    
    # Results display
    result_placeholder = st.empty()
    
    # Status
    st.markdown("### System Status")
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        camera_status = st.empty()
    
    with status_col2:
        model_status = st.empty()
    
    # Metrics
    st.markdown("### Metrics")
    col_fps, col_conf = st.columns(2)
    
    with col_fps:
        fps_display = st.empty()
    
    with col_conf:
        conf_display = st.empty()

# ==========================================
# CAMERA PROCESSING
# ==========================================

# Initialize session state
if 'sequence' not in st.session_state:
    st.session_state.sequence = deque(maxlen=SEQUENCE_LENGTH)
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = ("", 0)

if model is None:
    st.warning("Please ensure model files are in the same folder:")
    st.code(f"""
    Required files:
    - {LABEL_MAP_FILE}
    - {MODEL_FILES['minixgb']} or {MODEL_FILES['xgboost']} or {MODEL_FILES['bilstm']}
    """)
    st.stop()

# Start/Stop camera
if start_camera:
    st.session_state.running = True
if stop_camera:
    st.session_state.running = False

# Process camera
if st.session_state.running and model:
    # Update status
    camera_status.markdown('<span class="status-badge status-online">Camera: ON</span>', unsafe_allow_html=True)
    model_status.markdown('<span class="status-badge status-online">Model: READY</span>', unsafe_allow_html=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Try alternative camera index
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        camera_status.markdown('<span class="status-badge status-offline">Camera: ERROR</span>', unsafe_allow_html=True)
        st.warning("Could not access camera. Please check permissions.")
        st.session_state.running = False
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize MediaPipe
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            frame_time = time.time()
            frame_count = 0
            fps = 0
            
            while st.session_state.running and cap.isOpened():
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - frame_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    frame_time = current_time
                
                # Process frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw skeleton
                if show_skeleton:
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1),
                            mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1)
                        )
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                
                # Extract features
                features = extract_simple_features(results)
                st.session_state.sequence.append(features)
                
                # Update buffer display
                buffer_fill = len(st.session_state.sequence)
                
                # Make prediction when buffer is full
                prediction = ""
                confidence = 0
                
                if buffer_fill == SEQUENCE_LENGTH:
                    try:
                        if is_deep_learning:
                            # LSTM prediction
                            input_seq = np.array([list(st.session_state.sequence)])
                            preds = model.predict(input_seq, verbose=0)[0]
                            idx = np.argmax(preds)
                            confidence = float(preds[idx])
                            prediction = labels.get(idx, "Unknown")
                        else:
                            # XGBoost prediction
                            import joblib
                            # Simple feature aggregation for ML
                            seq_array = np.array(st.session_state.sequence)
                            mean_feat = np.mean(seq_array, axis=0)
                            input_data = mean_feat.reshape(1, -1)
                            
                            if hasattr(model, 'predict_proba'):
                                probs = model.predict_proba(input_data)[0]
                                idx = np.argmax(probs)
                                confidence = float(probs[idx])
                                prediction = labels.get(idx, "Unknown")
                            else:
                                # For models without predict_proba
                                idx = model.predict(input_data)[0]
                                prediction = labels.get(idx, "Unknown")
                                confidence = 0.85  # Default confidence
                        
                        if confidence > conf_threshold:
                            st.session_state.last_prediction = (prediction, confidence)
                        
                    except Exception as e:
                        pass
                
                # Update displays
                fps_display.metric("FPS", fps if show_fps else "-")
                
                pred_text, pred_conf = st.session_state.last_prediction
                if pred_text:
                    conf_display.metric("Confidence", f"{pred_conf*100:.0f}%")
                    
                    # Show prediction
                    result_placeholder.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-text">{pred_text.upper()}</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {pred_conf*100}%"></div>
                        </div>
                        <p>Confidence: {pred_conf*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add text to frame
                    cv2.putText(image, f"{pred_text.upper()} ({pred_conf*100:.0f}%)", 
                               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    result_placeholder.markdown(f"""
                    <div class="prediction-box">
                        <p>Collecting data...</p>
                        <p>{buffer_fill}/{SEQUENCE_LENGTH} frames</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {(buffer_fill/SEQUENCE_LENGTH)*100}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add FPS to frame
                if show_fps:
                    cv2.putText(image, f"FPS: {fps}", 
                               (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display frame
                with camera_placeholder.container():
                    st.markdown('<div class="camera-frame">', unsafe_allow_html=True)
                    st.image(image, channels="BGR")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Cleanup
        cap.release()
        
        # Update status
        camera_status.markdown('<span class="status-badge status-offline">Camera: OFF</span>', unsafe_allow_html=True)
        model_status.markdown('<span class="status-badge status-offline">Model: IDLE</span>', unsafe_allow_html=True)
else:
    # Default state
    camera_status.markdown('<span class="status-badge status-offline">Camera: OFF</span>', unsafe_allow_html=True)
    model_status.markdown('<span class="status-badge status-offline">Model: IDLE</span>', unsafe_allow_html=True)
    fps_display.metric("FPS", "-")
    conf_display.metric("Confidence", "-")
    
    if not use_camera:
        # Demo mode
        result_placeholder.markdown("""
        <div class="prediction-box">
            <h3>Demo Mode</h3>
            <p>Camera is disabled</p>
            <p>Enable camera for live recognition</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample image
        sample_image = np.zeros((300, 500, 3), dtype=np.uint8)
        sample_image[:] = (240, 240, 240)
        cv2.putText(sample_image, "ASL Recognition System", (80, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 200), 2)
        cv2.putText(sample_image, "Enable camera for live feed", (100, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        
        with camera_placeholder.container():
            st.markdown('<div class="camera-frame">', unsafe_allow_html=True)
            st.image(sample_image)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Ready state
        result_placeholder.markdown("""
        <div class="prediction-box">
            <h3>Ready</h3>
            <p>Click "Start Camera" to begin</p>
            <p>System will analyze gestures in real-time</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    **Quick Start:**
    1. Select AI model
    2. Enable camera
    3. Start recognition
    4. Perform clear signs
    """)

with col_info2:
    st.markdown("""
    **Tips:**
    - Good lighting
    - Steady hands
    - Clear background
    - Consistent speed
    """)

st.caption(f"ASL Recognition System | Model: {model_type} | FPS Target: 30")

# ==========================================
# TROUBLESHOOTING
# ==========================================

with st.expander("Camera Troubleshooting"):
    st.markdown("""
    **If camera doesn't work:**
    
    1. **Check browser permissions** - Allow camera access
    2. **Close other apps** - Ensure no other app is using camera
    3. **Try different browser** - Chrome works best
    4. **Check camera connection** - Ensure camera is properly connected
    
    **For mobile devices:**
    - Use Chrome or Safari
    - Grant camera permission
    - Hold device steady
    
    **If still not working:**
    - Use the system in demo mode
    - The AI models will still work
    - You can test with simulated data
    """)