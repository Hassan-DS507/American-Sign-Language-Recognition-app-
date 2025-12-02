import os
import sys
import json
import time
import warnings
import subprocess
from collections import deque

# ==========================================
# CONFIGURATION SECTION
# ==========================================

# System Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
warnings.filterwarnings('ignore')

# Application Constants
SEQUENCE_LENGTH = 50
CONFIDENCE_THRESHOLD = 0.75

# Model files (update these paths as needed)
MODEL_FILES = {
    'bilstm': 'bilstm_model_2.keras',
    'xgboost': 'xgboost_asl.pkl',
    'minixgb': 'xgb_model.pkl'
}

LABEL_MAP_FILE = 'label_map_2.json'

# ==========================================
# DEPENDENCY INSTALLATION
# ==========================================

def install_dependencies():
    """Install required packages if missing"""
    required_packages = [
        'streamlit==1.28.0',
        'opencv-python-headless==4.8.1.78',
        'mediapipe==0.10.8',
        'numpy==1.24.3',
        'joblib==1.3.2',
        'Pillow==10.1.0',
        'scikit-learn==1.3.0'
    ]
    
    for package in required_packages:
        package_name = package.split('==')[0]
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# ==========================================
# IMPORT LIBRARIES
# ==========================================

# Install dependencies first
install_dependencies()

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from PIL import Image

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Try to install TensorFlow
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.15.0"])
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        TF_AVAILABLE = True
    except:
        TF_AVAILABLE = False

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS (Professional Design)
# ==========================================

st.markdown("""
<style>
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .sub-title {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-top: 0.8rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    /* Prediction Display */
    .prediction-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border-left: 8px solid #667eea;
        min-height: 280px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    .prediction-word {
        font-size: 3rem;
        font-weight: 900;
        color: #2d3748;
        margin: 1.5rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .confidence-bar-container {
        width: 100%;
        height: 20px;
        background: #e2e8f0;
        border-radius: 10px;
        margin: 1.5rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #48bb78, #38a169);
        border-radius: 10px;
        transition: width 0.8s ease;
    }
    .confidence-text {
        font-size: 1.2rem;
        color: #4a5568;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    /* Metrics */
    .metric-box {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        border-top: 5px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #2d3748;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Status Indicators */
    .status-box {
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        text-align: center;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .status-active {
        background: #c6f6d5;
        color: #22543d;
        border-left: 4px solid #38a169;
    }
    .status-warning {
        background: #feebc8;
        color: #744210;
        border-left: 4px solid #dd6b20;
    }
    .status-inactive {
        background: #e2e8f0;
        color: #4a5568;
        border-left: 4px solid #a0aec0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
    }
    
    /* Camera Container */
    .camera-frame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        background: #000;
        border: 2px solid #e9ecef;
    }
    
    /* Progress Bar */
    .progress-container {
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    /* Error Messages */
    .error-box {
        background: #fed7d7;
        color: #742a2a;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #c53030;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #feebc8;
        color: #744210;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dd6b20;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================

st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 1.2rem; 
            border-radius: 12px; 
            color: white; 
            text-align: center;
            margin-bottom: 1.5rem;'>
    <h2 style='margin: 0; font-size: 1.5rem;'>CONTROL PANEL</h2>
</div>
""", unsafe_allow_html=True)

# Camera Mode Selection
st.sidebar.markdown("### OPERATION MODE")
camera_mode = st.sidebar.radio(
    "Select Input Source:",
    ("Live Camera", "Demo Mode")
)

# Model Selection
st.sidebar.markdown("### MODEL ARCHITECTURE")
model_type = st.sidebar.selectbox(
    "",
    ("Bi-LSTM (14 words)", 
     "XGBoost (14 words)",
     "Mini-XGBoost (5 words)")
)

# Configuration Settings
st.sidebar.markdown("### CONFIGURATION SETTINGS")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05
)
draw_landmarks = st.sidebar.checkbox("Show Skeleton Overlay", value=True)
show_fps = st.sidebar.checkbox("Display FPS Counter", value=True)

# Model Information
if "Bi-LSTM" in model_type:
    MODEL_FILE = MODEL_FILES['bilstm']
    IS_DEEP_LEARNING = True
elif "XGBoost" in model_type and "Mini" not in model_type:
    MODEL_FILE = MODEL_FILES['xgboost']
    IS_DEEP_LEARNING = False
else:  # Mini-XGBoost
    MODEL_FILE = MODEL_FILES['minixgb']
    IS_DEEP_LEARNING = False

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_dummy_frame():
    """Create a dummy frame for demo mode"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = [40, 40, 40]
    
    # Add title
    cv2.putText(frame, "ASL RECOGNITION SYSTEM", (80, 100), 
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (102, 126, 234), 2)
    
    # Add demo mode text
    cv2.putText(frame, "DEMO MODE ACTIVE", (180, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, "Connect camera for live feed", (150, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    
    # Draw a simple hand icon
    center_x, center_y = 320, 320
    # Palm
    cv2.circle(frame, (center_x, center_y), 60, (102, 126, 234), 2)
    # Fingers
    for angle in [-60, -30, 0, 30, 60]:
        rad = np.deg2rad(angle)
        x = int(center_x + 80 * np.sin(rad))
        y = int(center_y - 80 * np.cos(rad))
        cv2.line(frame, (center_x, center_y), (x, y), (102, 126, 234), 2)
    
    return frame

# ==========================================
# PREPROCESSING FUNCTIONS
# ==========================================

def normalize_hand(pts):
    """Normalize hand landmarks relative to wrist"""
    ref = pts[0].copy()
    scale = np.linalg.norm(pts[9] - ref)
    if scale < 1e-6: 
        scale = 1.0
    return (pts - ref) / scale

def compute_torso_stats(pose_landmarks):
    """Compute torso center and scale for normalization"""
    torso_center = np.array([0.5, 0.5], dtype=np.float32)
    torso_scale = 1.0
    
    if pose_landmarks:
        try:
            def get_xy(idx): 
                return np.array([
                    pose_landmarks.landmark[idx].x, 
                    pose_landmarks.landmark[idx].y
                ], dtype=np.float32)
            
            left_sh = get_xy(11)
            right_sh = get_xy(12)
            left_hip = get_xy(23)
            right_hip = get_xy(24)
            
            shoulder_center = (left_sh + right_sh) / 2.0
            hip_center = (left_hip + right_hip) / 2.0
            torso_center = (shoulder_center + hip_center) / 2.0
            
            shoulder_dist = np.linalg.norm(left_sh - right_sh)
            hip_dist = np.linalg.norm(left_hip - right_hip)
            torso_scale = max(shoulder_dist, hip_dist, 1e-6)
        except:
            pass
    
    return torso_center, float(torso_scale)

def extract_features(results):
    """Extract features from MediaPipe results"""
    feat = np.zeros(198, dtype=np.float32)
    
    # Pose landmarks
    torso_center = np.array([0.5, 0.5], dtype=np.float32)
    torso_scale = 1.0
    
    if results.pose_landmarks:
        torso_center, torso_scale = compute_torso_stats(results.pose_landmarks)
        pose_xy = np.array([
            [lm.x, lm.y] for lm in results.pose_landmarks.landmark
        ], dtype=np.float32)
        pose_norm = (pose_xy - torso_center[None, :]) / torso_scale
        feat[0:66] = pose_norm.flatten()

    # Left hand
    if results.left_hand_landmarks:
        l_pts = np.array([
            [lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark
        ], dtype=np.float32)
        feat[66:129] = normalize_hand(l_pts)[:, :3].flatten()
        wrist_rel = l_pts[0].copy()
        wrist_rel[:2] = (wrist_rel[:2] - torso_center) / torso_scale
        wrist_rel[2] /= torso_scale
        feat[129:132] = wrist_rel

    # Right hand
    if results.right_hand_landmarks:
        r_pts = np.array([
            [lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark
        ], dtype=np.float32)
        feat[132:195] = normalize_hand(r_pts)[:, :3].flatten()
        wrist_rel = r_pts[0].copy()
        wrist_rel[:2] = (wrist_rel[:2] - torso_center) / torso_scale
        wrist_rel[2] /= torso_scale
        feat[195:198] = wrist_rel
    
    return feat

def feature_engineering_for_ml(sequence_buffer):
    """Convert sequence to statistical features for ML models"""
    raw_data = np.array(sequence_buffer, dtype=np.float32)
    mean_feat = np.mean(raw_data, axis=0)
    std_feat = np.std(raw_data, axis=0)
    max_feat = np.max(raw_data, axis=0)
    velocity = np.diff(raw_data, axis=0)
    mean_vel = np.mean(velocity, axis=0)
    final_vec = np.concatenate([mean_feat, std_feat, max_feat, mean_vel])
    return final_vec.reshape(1, -1)

# ==========================================
# MODEL LOADING
# ==========================================

@st.cache_resource
def load_models_and_labels(_model_file, is_dl):
    """Load model and labels with caching"""
    model = None
    actions = {}
    
    try:
        # Load model
        if is_dl:
            if TF_AVAILABLE:
                try:
                    model = tf.keras.models.load_model(_model_file, compile=False)
                    model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                except:
                    return None, None
            else:
                return None, None
        else:
            model = joblib.load(_model_file)
        
        # Load labels
        try:
            with open(LABEL_MAP_FILE, 'r') as f:
                label_map = json.load(f)
            
            if isinstance(label_map, dict):
                actions = {v: k for k, v in label_map.items()}
            else:
                actions = {i: label for i, label in enumerate(label_map)}
        except FileNotFoundError:
            if "Mini" in model_type:
                actions = {0: 'banana', 1: 'jacket', 2: 'cry', 3: 'catch', 4: 'pop'}
            else:
                default_labels = ['hello', 'thanks', 'yes', 'no', 'please', 
                                'sorry', 'help', 'water', 'food', 'bathroom',
                                'medical', 'emergency', 'love', 'family']
                actions = {i: label for i, label in enumerate(default_labels)}
        
        return model, actions
        
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None

# Load model based on selection
if "Mini" in model_type:
    actions = {0: 'banana', 1: 'jacket', 2: 'cry', 3: 'catch', 4: 'pop'}
    try:
        model = joblib.load(MODEL_FILE)
        MODEL_LOADED = True
    except:
        MODEL_LOADED = False
        model = None
else:
    model, actions = load_models_and_labels(MODEL_FILE, IS_DEEP_LEARNING)
    MODEL_LOADED = model is not None

# ==========================================
# MAIN APPLICATION UI
# ==========================================

# Header
st.markdown("""
<div class='main-header'>
    <h1 class='main-title'>ASL Recognition System</h1>
    <p class='sub-title'>Real-time American Sign Language Translation with AI</p>
</div>
""", unsafe_allow_html=True)

# Check if model loaded successfully
if not MODEL_LOADED:
    st.markdown("""
    <div class='error-box'>
        <h3 style='margin: 0;'>MODEL LOADING REQUIRED</h3>
        <p>Please ensure these files are in the same folder:</p>
        <ul>
            <li><code>{}</code> - Selected model file</li>
            <li><code>{}</code> - Label mapping (for full models)</li>
        </ul>
    </div>
    """.format(MODEL_FILE, LABEL_MAP_FILE), unsafe_allow_html=True)
    
    # Show file status
    st.markdown("### FILE STATUS")
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(MODEL_FILE):
            st.success(f"✓ {MODEL_FILE}")
        else:
            st.error(f"✗ {MODEL_FILE}")
    
    with col2:
        if os.path.exists(LABEL_MAP_FILE):
            st.success(f"✓ {LABEL_MAP_FILE}")
        else:
            st.warning(f"⚠ {LABEL_MAP_FILE}")
    
    st.stop()

# Create main layout
col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.markdown("### LIVE FEED")
    
    # Show current mode
    if camera_mode == "Live Camera":
        st.info("**Mode:** Live Camera - Requires camera access")
    else:
        st.info("**Mode:** Demo - No camera required")
    
    # Start button
    if st.button("START RECOGNITION", key="start_button", use_container_width=True):
        st.session_state.running = True
    if st.button("STOP", key="stop_button", use_container_width=True):
        st.session_state.running = False
        st.rerun()
    
    # Camera feed container
    camera_placeholder = st.empty()

with col2:
    st.markdown("### RECOGNITION RESULTS")
    
    # Results container
    result_placeholder = st.empty()
    
    # System status
    st.markdown("#### SYSTEM STATUS")
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        status_camera = st.empty()
        status_camera.markdown("""
        <div class='status-box status-inactive'>
            Camera: Offline
        </div>
        """, unsafe_allow_html=True)
    
    with status_col2:
        status_model = st.empty()
        status_model.markdown("""
        <div class='status-box status-inactive'>
            Model: Idle
        </div>
        """, unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("#### PERFORMANCE")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        metric_fps = st.empty()
        metric_fps.markdown("""
        <div class='metric-box'>
            <div class='metric-value'>0</div>
            <div class='metric-label'>FPS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        metric_conf = st.empty()
        metric_conf.markdown("""
        <div class='metric-box'>
            <div class='metric-value'>0%</div>
            <div class='metric-label'>Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        metric_buffer = st.empty()
        metric_buffer.markdown("""
        <div class='metric-box'>
            <div class='metric-value'>0%</div>
            <div class='metric-label'>Buffer</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# RECOGNITION LOOP
# ==========================================

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'sequence' not in st.session_state:
    st.session_state.sequence = deque(maxlen=SEQUENCE_LENGTH)
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'fps_buffer' not in st.session_state:
    st.session_state.fps_buffer = deque(maxlen=30)

if st.session_state.running:
    # Update status
    status_camera.markdown(f"""
    <div class='status-box status-active'>
        Camera: {camera_mode}
    </div>
    """, unsafe_allow_html=True)
    
    status_model.markdown("""
    <div class='status-box status-active'>
        Model: Processing
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize camera if in live mode
    cap = None
    if camera_mode == "Live Camera":
        try:
            # Try different camera indices
            for idx in [0, 1, 2]:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    break
                cap.release()
        except:
            cap = None
    
    # Initialize MediaPipe
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        frame_placeholder = camera_placeholder.empty()
        
        # Main processing loop
        try:
            while st.session_state.running:
                # Calculate FPS
                st.session_state.frame_count += 1
                elapsed_time = time.time() - st.session_state.start_time
                
                if elapsed_time > 0:
                    current_fps = st.session_state.frame_count / elapsed_time
                    st.session_state.fps_buffer.append(current_fps)
                    avg_fps = np.mean(st.session_state.fps_buffer) if st.session_state.fps_buffer else 0
                    
                    # Update FPS display
                    metric_fps.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-value'>{int(avg_fps)}</div>
                        <div class='metric-label'>FPS</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Get frame
                if camera_mode == "Live Camera" and cap and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = holistic.process(image)
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        # Draw landmarks
                        if draw_landmarks:
                            if results.pose_landmarks:
                                mp_drawing.draw_landmarks(
                                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(100, 117, 186), thickness=2),
                                    mp_drawing.DrawingSpec(color=(100, 117, 186), thickness=2)
                                )
                            if results.left_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2),
                                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2)
                                )
                            if results.right_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                                )
                        
                        # Extract features
                        keypoints = extract_features(results)
                        st.session_state.sequence.append(keypoints)
                    else:
                        image = create_dummy_frame()
                else:
                    # Demo mode
                    image = create_dummy_frame()
                    results = None
                    
                    # Simulate feature extraction for demo
                    if st.session_state.frame_count % 10 == 0:
                        dummy_features = np.random.randn(198).astype(np.float32)
                        st.session_state.sequence.append(dummy_features)
                
                # Update buffer display
                buffer_fill = min(100, int((len(st.session_state.sequence) / SEQUENCE_LENGTH) * 100))
                metric_buffer.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{buffer_fill}%</div>
                    <div class='metric-label'>Buffer</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Make prediction when buffer is full
                current_prediction = None
                current_confidence = 0
                
                if len(st.session_state.sequence) == SEQUENCE_LENGTH:
                    try:
                        if camera_mode == "Demo Mode":
                            # Demo predictions
                            demo_words = ['hello', 'thanks', 'yes', 'no', 'please', 'help', 'water']
                            current_prediction = np.random.choice(demo_words)
                            current_confidence = np.random.uniform(0.7, 0.95)
                        elif IS_DEEP_LEARNING:
                            input_data = np.expand_dims(list(st.session_state.sequence), axis=0).astype(np.float32)
                            predictions = model.predict(input_data, verbose=0)[0]
                            best_idx = np.argmax(predictions)
                            current_confidence = predictions[best_idx]
                            current_prediction = actions.get(best_idx, "Unknown")
                        else:
                            input_data = feature_engineering_for_ml(st.session_state.sequence)
                            probabilities = model.predict_proba(input_data)[0]
                            best_idx = np.argmax(probabilities)
                            current_confidence = probabilities[best_idx]
                            current_prediction = actions.get(best_idx, "Unknown")
                        
                        if current_confidence > conf_threshold:
                            st.session_state.predictions.append((current_prediction, current_confidence))
                            if len(st.session_state.predictions) > 5:
                                st.session_state.predictions.pop(0)
                            
                            metric_conf.markdown(f"""
                            <div class='metric-box'>
                                <div class='metric-value'>{int(current_confidence*100)}%</div>
                                <div class='metric-label'>Confidence</div>
                            </div>
                            """, unsafe_allow_html=True)
                    except:
                        pass
                
                # Display prediction
                if current_prediction:
                    result_placeholder.markdown(f"""
                    <div class='prediction-container'>
                        <h3 style='color: #4a5568; margin: 0;'>DETECTED SIGN</h3>
                        <div class='prediction-word'>{current_prediction.upper()}</div>
                        <div class='confidence-bar-container'>
                            <div class='confidence-fill' style='width: {int(current_confidence*100)}%'></div>
                        </div>
                        <p class='confidence-text'>Confidence: {int(current_confidence*100)}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add overlay to frame
                    if camera_mode == "Live Camera" and 'frame' in locals():
                        cv2.putText(image, f"Sign: {current_prediction}", 
                                  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    result_placeholder.markdown(f"""
                    <div class='prediction-container'>
                        <h3 style='color: #4a5568; margin: 0;'>ANALYZING GESTURE</h3>
                        <div style='font-size: 3rem; color: #a0aec0; margin: 1.5rem;'>ASL</div>
                        <p style='color: #718096;'>Processing: {len(st.session_state.sequence)}/{SEQUENCE_LENGTH}</p>
                        <div class='progress-container'>
                            <div class='progress-fill' style='width: {buffer_fill}%'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add FPS counter
                if show_fps and avg_fps > 0 and camera_mode == "Live Camera":
                    cv2.putText(image, f"FPS: {int(avg_fps)}", 
                              (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display frame - FIXED: No use_container_width parameter
                try:
                    frame_placeholder.image(image, channels="BGR")
                except:
                    # Fallback for Streamlit compatibility
                    frame_placeholder.image(image)
                
                # Small delay to prevent freezing
                time.sleep(0.03)
                
        except Exception as e:
            st.error(f"Processing error: {str(e)[:100]}")
        finally:
            if cap:
                cap.release()
            
            # Reset status
            status_camera.markdown("""
            <div class='status-box status-inactive'>
                Camera: Offline
            </div>
            """, unsafe_allow_html=True)
            
            status_model.markdown("""
            <div class='status-box status-inactive'>
                Model: Idle
            </div>
            """, unsafe_allow_html=True)
            
            # Show summary
            if st.session_state.predictions:
                st.markdown("---")
                st.markdown("### SESSION SUMMARY")
                cols = st.columns(min(5, len(st.session_state.predictions)))
                for idx, (pred, conf) in enumerate(st.session_state.predictions[:5]):
                    with cols[idx % len(cols)]:
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div style='color: #667eea; font-size: 1.2rem; font-weight: 700;'>{pred.upper()}</div>
                            <div style='color: #718096; font-size: 0.9rem;'>{int(conf*100)}%</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Clear session state
            st.session_state.sequence.clear()
            st.session_state.predictions.clear()
            st.session_state.frame_count = 0
            st.session_state.start_time = time.time()
else:
    # Welcome state
    result_placeholder.markdown("""
    <div class='prediction-container'>
        <h2 style='color: #2d3748;'>READY FOR RECOGNITION</h2>
        <p style='color: #718096; margin: 1rem 0;'>
            Select mode and click START to begin ASL translation
        </p>
        <div style='font-size: 4rem; color: #667eea; margin: 1rem;'>ASL</div>
        <p style='color: #718096; font-size: 0.9rem;'>
            Choose between Live Camera or Demo Mode
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("""
    ### GETTING STARTED
    1. Select operation mode
    2. Choose AI model
    3. Adjust settings
    4. Click Start
    """)

with col_f2:
    st.markdown("""
    ### TIPS
    - Good lighting for accuracy
    - Clear background
    - Steady hand movements
    - Live mode needs camera
    """)

with col_f3:
    st.markdown("""
    ### SYSTEM
    - Sequence: 50 frames
    - Real-time processing
    - Multiple AI models
    - Confidence scoring
    """)

st.markdown("---")
st.caption(f"ASL Recognition System | Model: {model_type} | Mode: {camera_mode}")

# ==========================================
# HELP SECTION
# ==========================================

with st.expander("HELP & TROUBLESHOOTING"):
    st.markdown("""
    ### CAMERA ISSUES
    
    **Live Camera not working:**
    1. Ensure camera is connected
    2. Grant camera permission in browser
    3. Try a different browser (Chrome recommended)
    4. Close other apps using camera
    
    **For Demo Mode:**
    - No camera required
    - Simulated predictions
    - Perfect for testing
    
    ### MODEL FILES
    
    **Required files in same folder:**
    - `app.py` - This application
    - `label_map_2.json` - Label definitions
    - One of these model files:
        - `bilstm_model_2.keras` (Bi-LSTM)
        - `xgboost_asl.pkl` (XGBoost)
        - `xgb_model.pkl` (Mini)
    
    ### PERFORMANCE
    
    **For best results:**
    - Use XGBoost for speed
    - Use Bi-LSTM for accuracy
    - Adjust confidence threshold
    - Ensure good lighting
    """)