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
    print("TensorFlow not available, Bi-LSTM model will not work")
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
# CUSTOM CSS (Professional Design - No Emojis)
# ==========================================

st.markdown("""
<style>
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .sub-title {
        font-size: 1.4rem;
        opacity: 0.95;
        margin-top: 0.8rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.8rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* Prediction Display */
    .prediction-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        border-left: 8px solid #667eea;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    .prediction-word {
        font-size: 3.5rem;
        font-weight: 900;
        color: #2d3748;
        margin: 1.5rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .confidence-bar-container {
        width: 100%;
        height: 25px;
        background: #e2e8f0;
        border-radius: 12px;
        margin: 1.5rem 0;
        overflow: hidden;
        position: relative;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #48bb78, #38a169, #2f855a);
        border-radius: 12px;
        transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .confidence-text {
        font-size: 1.3rem;
        color: #4a5568;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    /* Metrics */
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border-top: 5px solid #667eea;
        margin: 0.8rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2d3748;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Status Indicators */
    .status-box {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }
    .status-active {
        background: linear-gradient(135deg, #c6f6d5, #9ae6b4);
        color: #22543d;
        border-left: 5px solid #38a169;
    }
    .status-warning {
        background: linear-gradient(135deg, #feebc8, #fbd38d);
        color: #744210;
        border-left: 5px solid #dd6b20;
    }
    .status-inactive {
        background: linear-gradient(135deg, #e2e8f0, #cbd5e0);
        color: #4a5568;
        border-left: 5px solid #a0aec0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Camera Container */
    .camera-frame {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        background: #000;
        position: relative;
        border: 3px solid #e9ecef;
    }
    
    /* Progress Bar */
    .progress-container {
        width: 100%;
        height: 10px;
        background: #e2e8f0;
        border-radius: 5px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    
    /* Icons */
    .icon-camera:before { content: "📷 "; }
    .icon-model:before { content: "🤖 "; }
    .icon-fps:before { content: "⚡ "; }
    .icon-confidence:before { content: "🎯 "; }
    .icon-buffer:before { content: "📊 "; }
    .icon-start:before { content: "🚀 "; }
    .icon-stop:before { content: "⏹️ "; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================

st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 1.5rem; 
            border-radius: 15px; 
            color: white; 
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);'>
    <h2 style='margin: 0; font-size: 1.8rem;'>CONTROL PANEL</h2>
</div>
""", unsafe_allow_html=True)

# Model Selection
st.sidebar.markdown("### MODEL ARCHITECTURE")
model_type = st.sidebar.selectbox(
    "",
    ("Bi-LSTM (14 words)", 
     "XGBoost (14 words)",
     "Mini-XGBoost (5 words)"),
    help="Select the AI model for recognition"
)

# Configuration Settings
st.sidebar.markdown("### CONFIGURATION SETTINGS")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05,
    help="Minimum confidence level for predictions"
)
draw_landmarks = st.sidebar.checkbox("Show Skeleton Overlay", value=True,
                                     help="Display MediaPipe skeleton on camera feed")
show_fps = st.sidebar.checkbox("Display FPS Counter", value=True,
                               help="Show frames per second counter")

# Model Information
st.sidebar.markdown("### MODEL INFORMATION")

# Set model file based on selection
if "Bi-LSTM" in model_type:
    MODEL_FILE = MODEL_FILES['bilstm']
    IS_DEEP_LEARNING = True
    if TF_AVAILABLE:
        st.sidebar.markdown("""
        <div class='card'>
            <h4 style='color: #667eea;'>Bi-LSTM Neural Network</h4>
            <p><strong>Architecture:</strong> Bidirectional LSTM with Hybrid Pooling</p>
            <p><strong>Accuracy:</strong> 87.86%</p>
            <p><strong>Best For:</strong> Complex temporal patterns</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.error("TensorFlow not available. Bi-LSTM will not work.")
elif "XGBoost" in model_type and "Mini" not in model_type:
    MODEL_FILE = MODEL_FILES['xgboost']
    IS_DEEP_LEARNING = False
    st.sidebar.markdown("""
    <div class='card'>
        <h4 style='color: #667eea;'>XGBoost Classifier</h4>
        <p><strong>Architecture:</strong> Gradient Boosting Decision Trees</p>
        <p><strong>Accuracy:</strong> 79.56%</p>
        <p><strong>Best For:</strong> Fast inference, distinct poses</p>
    </div>
    """, unsafe_allow_html=True)
else:  # Mini-XGBoost
    MODEL_FILE = MODEL_FILES['minixgb']
    IS_DEEP_LEARNING = False
    st.sidebar.markdown("""
    <div class='card'>
        <h4 style='color: #667eea;'>Mini XGBoost (Demo)</h4>
        <p><strong>Vocabulary:</strong> 5 basic signs</p>
        <p><strong>Purpose:</strong> Quick testing & demonstration</p>
        <p><strong>Best For:</strong> Rapid prototyping</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# KERAS MODEL LOADING FIX
# ==========================================

def fix_keras_model(model_path):
    """
    Fix for Keras 3.x compatibility issues
    Creates a compatible model based on your architecture
    """
    if not TF_AVAILABLE:
        return None
    
    try:
        # First try direct loading
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        try:
            # Load with compile=False and custom objects
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={
                    'AdamW': tf.keras.optimizers.legacy.AdamW
                }
            )
            
            # Recompile the model
            model.compile(
                optimizer=tf.keras.optimizers.legacy.AdamW(learning_rate=0.001, weight_decay=1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            return model
        except Exception as e:
            st.error(f"Failed to load Keras model: {str(e)[:100]}")
            return None

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
                model = fix_keras_model(_model_file)
                if model is None:
                    return None, None
            else:
                st.error("TensorFlow not available for deep learning model")
                return None, None
        else:
            # Load ML model
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
            # If label file not found, use default based on model type
            if "Mini" in model_type:
                actions = {0: 'banana', 1: 'jacket', 2: 'cry', 3: 'catch', 4: 'pop'}
            else:
                # Default labels for 14 classes
                default_labels = ['hello', 'thanks', 'yes', 'no', 'please', 
                                'sorry', 'help', 'water', 'food', 'bathroom',
                                'medical', 'emergency', 'love', 'family']
                actions = {i: label for i, label in enumerate(default_labels)}
        
        return model, actions
        
    except FileNotFoundError:
        st.error(f"Model file '{_model_file}' not found")
        return None, None
    except Exception as e:
        st.error(f"Error loading resources: {str(e)[:100]}")
        return None, None

# Load model based on selection
if "Mini" in model_type:
    # For mini model, use hardcoded labels
    actions = {0: 'banana', 1: 'jacket', 2: 'cry', 3: 'catch', 4: 'pop'}
    try:
        model = joblib.load(MODEL_FILE)
        MODEL_LOADED = True
    except:
        st.error(f"Failed to load model: {MODEL_FILE}")
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
    st.error(f"""
    ### MODEL LOADING REQUIRED
    
    **Please ensure these files are in the same folder as app.py:**
    
    **For Mini-XGBoost (5 words):**
    - `{MODEL_FILES['minixgb']}` - Mini model file
    
    **For Bi-LSTM (14 words):**
    - `{MODEL_FILES['bilstm']}` - Deep learning model
    - `{LABEL_MAP_FILE}` - Label mapping JSON
    
    **For XGBoost (14 words):**
    - `{MODEL_FILES['xgboost']}` - Machine learning model  
    - `{LABEL_MAP_FILE}` - Label mapping JSON
    
    **Current selection:** {model_type}
    **Expected file:** `{MODEL_FILE}`
    """)
    
    # Show file existence check
    st.markdown("### FILE STATUS")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if os.path.exists(MODEL_FILE):
            st.success(f"{MODEL_FILE} - Found")
        else:
            st.error(f"{MODEL_FILE} - Missing")
    
    with col2:
        if os.path.exists(LABEL_MAP_FILE):
            st.success(f"{LABEL_MAP_FILE} - Found")
        else:
            st.warning(f"{LABEL_MAP_FILE} - Missing")
    
    with col3:
        if TF_AVAILABLE:
            st.success("TensorFlow - Installed")
        else:
            st.error("TensorFlow - Not installed")
    
    st.stop()

# Create main layout
col1, col2 = st.columns([0.65, 0.35])

with col1:
    st.markdown("### LIVE CAMERA FEED")
    
    # Camera controls
    col_start, col_stop = st.columns(2)
    with col_start:
        run_camera = st.button("START RECOGNITION", use_container_width=True)
    with col_stop:
        stop_button = st.button("STOP SYSTEM", use_container_width=True)
        if stop_button:
            run_camera = False
            st.rerun()
    
    # Camera feed container
    camera_container = st.empty()

with col2:
    st.markdown("### RECOGNITION RESULTS")
    
    # Results container
    result_container = st.empty()
    
    # System status
    st.markdown("#### SYSTEM STATUS")
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        camera_status = st.empty()
        camera_status.markdown("""
        <div class='status-box status-inactive'>
            Camera: Offline
        </div>
        """, unsafe_allow_html=True)
    
    with status_col2:
        model_status = st.empty()
        model_status.markdown("""
        <div class='status-box status-inactive'>
            Model: Idle
        </div>
        """, unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("#### PERFORMANCE METRICS")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        fps_display = st.empty()
        fps_display.markdown("""
        <div class='metric-box'>
            <div class='metric-value'>0</div>
            <div class='metric-label'>FPS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        conf_display = st.empty()
        conf_display.markdown("""
        <div class='metric-box'>
            <div class='metric-value'>0%</div>
            <div class='metric-label'>Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        buffer_display = st.empty()
        buffer_display.markdown("""
        <div class='metric-box'>
            <div class='metric-value'>0%</div>
            <div class='metric-label'>Buffer</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# RECOGNITION LOOP
# ==========================================

sequence = deque(maxlen=SEQUENCE_LENGTH)
prediction_history = []
fps_buffer = deque(maxlen=30)
avg_fps = 0

if run_camera:
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open camera. Please check camera connection.")
        st.stop()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Update status
    camera_status.markdown("""
    <div class='status-box status-active'>
        Camera: Active
    </div>
    """, unsafe_allow_html=True)
    
    model_status.markdown("""
    <div class='status-box status-active'>
        Model: Processing
    </div>
    """, unsafe_allow_html=True)
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        
        frame_count = 0
        start_time = time.time()
        frame_placeholder = camera_container.empty()
        
        while cap.isOpened() and run_camera:
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time > 0:
                current_fps = frame_count / elapsed_time
                fps_buffer.append(current_fps)
                avg_fps = np.mean(fps_buffer) if fps_buffer else 0
                
                # Update FPS display
                fps_display.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{int(avg_fps)}</div>
                    <div class='metric-label'>FPS</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame from camera")
                break
            
            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks if enabled
            if draw_landmarks and results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(100, 117, 186), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(100, 117, 186), thickness=2, circle_radius=3)
                )
            
            if draw_landmarks and results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=3, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=3, circle_radius=4)
                )
            
            if draw_landmarks and results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=3, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=3, circle_radius=4)
                )
            
            # Extract features
            keypoints = extract_features(results)
            sequence.append(keypoints)
            
            # Update buffer display
            buffer_fill = min(100, int((len(sequence) / SEQUENCE_LENGTH) * 100))
            buffer_display.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{buffer_fill}%</div>
                <div class='metric-label'>Buffer</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Perform inference when buffer is full
            current_prediction = None
            current_confidence = 0
            
            if len(sequence) == SEQUENCE_LENGTH:
                try:
                    if IS_DEEP_LEARNING:
                        # Bi-LSTM inference
                        input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
                        predictions = model.predict(input_data, verbose=0)[0]
                        best_idx = np.argmax(predictions)
                        confidence = predictions[best_idx]
                    else:
                        # XGBoost inference
                        input_data = feature_engineering_for_ml(sequence)
                        probabilities = model.predict_proba(input_data)[0]
                        best_idx = np.argmax(probabilities)
                        confidence = probabilities[best_idx]
                    
                    # Only update if confidence meets threshold
                    if confidence > conf_threshold:
                        current_prediction = actions.get(best_idx, "Unknown")
                        current_confidence = confidence
                        
                        # Add to history
                        prediction_history.append((current_prediction, current_confidence))
                        if len(prediction_history) > 5:
                            prediction_history.pop(0)
                        
                        # Update confidence display
                        conf_display.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-value'>{int(current_confidence*100)}%</div>
                            <div class='metric-label'>Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    pass
            
            # Display prediction
            if current_prediction:
                result_container.markdown(f"""
                <div class='prediction-container'>
                    <h3 style='color: #4a5568; margin: 0;'>DETECTED SIGN</h3>
                    <div class='prediction-word'>{current_prediction.upper()}</div>
                    <div class='confidence-bar-container'>
                        <div class='confidence-fill' style='width: {int(current_confidence*100)}%'></div>
                    </div>
                    <p class='confidence-text'>Confidence: {int(current_confidence*100)}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add overlay to video frame
                cv2.rectangle(image, (0, 0), (640, 80), (0, 0, 0), -1)
                cv2.putText(
                    image, f"Sign: {current_prediction}", 
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3
                )
                cv2.putText(
                    image, f"Confidence: {int(current_confidence*100)}%", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2
                )
            else:
                # Show waiting state
                result_container.markdown(f"""
                <div class='prediction-container'>
                    <h3 style='color: #4a5568; margin: 0;'>ANALYZING GESTURE</h3>
                    <div style='font-size: 3.5rem; color: #a0aec0; margin: 2rem;'>---</div>
                    <p style='color: #718096; font-size: 1.1rem;'>
                        Processing {len(sequence)}/{SEQUENCE_LENGTH} frames
                    </p>
                    <div class='progress-container'>
                        <div class='progress-fill' style='width: {buffer_fill}%'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add FPS counter to frame if enabled
            if show_fps and avg_fps > 0:
                cv2.putText(
                    image, f"FPS: {int(avg_fps)}", 
                    (520, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
                )
            
            # Display frame with camera frame styling
            with camera_container.container():
                st.markdown('<div class="camera-frame">', unsafe_allow_html=True)
                frame_placeholder.image(image, channels="BGR", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Cleanup
    cap.release()
    
    # Final status update
    camera_status.markdown("""
    <div class='status-box status-inactive'>
        Camera: Offline
    </div>
    """, unsafe_allow_html=True)
    
    model_status.markdown("""
    <div class='status-box status-inactive'>
        Model: Idle
    </div>
    """, unsafe_allow_html=True)
    
    # Show session summary
    if prediction_history:
        st.markdown("---")
        st.markdown("### SESSION SUMMARY")
        cols = st.columns(len(prediction_history))
        for idx, (pred, conf) in enumerate(prediction_history):
            with cols[idx]:
                color = "#667eea" if conf > 0.8 else "#dd6b20" if conf > 0.6 else "#a0aec0"
                st.markdown(f"""
                <div class='metric-box'>
                    <h4 style='color: {color};'>{pred.upper()}</h4>
                    <p style='color: #718096; font-size: 0.9rem; font-weight: 600;'>
                        {int(conf*100)}% confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
else:
    # Welcome state
    result_container.markdown("""
    <div class='prediction-container'>
        <h2 style='color: #2d3748;'>READY FOR RECOGNITION</h2>
        <p style='color: #718096; margin: 1rem 0; font-size: 1.2rem;'>
            Click <strong style='color: #667eea;'>START RECOGNITION</strong> to begin real-time ASL translation
        </p>
        <div style='font-size: 5rem; color: #667eea; margin: 1.5rem;'>ASL</div>
        <p style='color: #718096; font-size: 1rem;'>
            Ensure proper lighting and clear view of your hands
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    ### QUICK START
    1. Select model type
    2. Adjust confidence threshold
    3. Click Start Recognition
    4. Perform signs clearly
    """)

with footer_col2:
    st.markdown("""
    ### TIPS FOR ACCURACY
    - Good lighting is essential
    - Keep hands in frame
    - Clear background
    - Consistent signing speed
    """)

with footer_col3:
    st.markdown("""
    ### SYSTEM INFO
    - Sequence Length: 50 frames
    - Processing: Real-time
    - Models: 3 AI architectures
    - Output: Word + Confidence
    """)

st.markdown("---")
st.caption(f"ASL Recognition System v2.0 | Model: {model_type}")

# ==========================================
# TROUBLESHOOTING SECTION
# ==========================================

with st.expander("TROUBLESHOOTING"):
    st.markdown("""
    ### COMMON ISSUES
    
    **1. Camera not working:**
    - Check camera permissions
    - Ensure no other app is using the camera
    - Try a different browser
    
    **2. Model not loading:**
    - Ensure model files are in the same folder as app.py
    - Check file names exactly match:
        - `bilstm_model_2.keras`
        - `xgboost_asl.pkl` 
        - `xgb_model.pkl`
        - `label_map_2.json`
    
    **3. Low FPS:**
    - Disable skeleton overlay
    - Reduce camera resolution in code
    - Use XGBoost instead of Bi-LSTM
    
    **4. Low accuracy:**
    - Increase confidence threshold
    - Ensure good lighting
    - Clear, plain background
    - Hold signs for 1-2 seconds
    """)