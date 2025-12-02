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
warnings.filterwarnings('ignore')

# Application Constants
SEQUENCE_LENGTH = 50
CONFIDENCE_THRESHOLD = 0.75
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
        'Pillow==10.1.0'
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

# Now import all libraries
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

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .sub-title {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        border-left: 6px solid #667eea;
        min-height: 250px;
    }
    .prediction-word {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2d3748;
        margin: 1rem 0;
        text-transform: uppercase;
    }
    .confidence-bar {
        width: 100%;
        height: 20px;
        background: #e2e8f0;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #48bb78, #38a169);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-top: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
    }
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin: 0.2rem 0;
    }
    .status-active { background: #c6f6d5; color: #22543d; }
    .status-inactive { background: #e2e8f0; color: #4a5568; }
    .status-warning { background: #feebc8; color: #744210; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================

st.sidebar.markdown("""
<div class='metric-box'>
    <h3 style='margin:0; color:#667eea;'>ASL Control Panel</h3>
</div>
""", unsafe_allow_html=True)

# Model Selection
st.sidebar.markdown("### Model Architecture")
model_type = st.sidebar.selectbox(
    "Select Model Type:",
    ("Bi-LSTM (14 words)", 
     "XGBoost (14 words)",
     "Mini-XGBoost (5 words)")
)

# Configuration Settings
st.sidebar.markdown("### Configuration")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05
)
draw_landmarks = st.sidebar.checkbox("Show Skeleton Overlay", value=True)
show_fps = st.sidebar.checkbox("Display FPS Counter", value=True)

# Model Information
st.sidebar.markdown("### Model Information")

if "Bi-LSTM" in model_type:
    MODEL_FILE = MODEL_FILES['bilstm']
    IS_DEEP_LEARNING = True
    if not TF_AVAILABLE:
        st.sidebar.error("TensorFlow not available for Bi-LSTM model")
else:
    IS_DEEP_LEARNING = False
    if "Mini" in model_type:
        MODEL_FILE = MODEL_FILES['minixgb']
    else:
        MODEL_FILE = MODEL_FILES['xgboost']

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
        # Load labels from label_map_2.json
        with open(LABEL_MAP_FILE, 'r') as f:
            label_map = json.load(f)
        
        if isinstance(label_map, dict):
            actions = {v: k for k, v in label_map.items()}
        else:
            actions = {i: label for i, label in enumerate(label_map)}
        
        # Load model
        if is_dl:
            if TF_AVAILABLE:
                try:
                    # Try loading the Keras model
                    model = tf.keras.models.load_model(_model_file, compile=False)
                    
                    # Recompile with simple optimizer
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                except Exception as e:
                    st.error(f"Error loading Keras model: {str(e)[:100]}")
                    return None, None
            else:
                st.error("TensorFlow not available for deep learning model")
                return None, None
        else:
            # Load ML model
            model = joblib.load(_model_file)
            
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
    except:
        st.error(f"Failed to load model: {MODEL_FILE}")
        model = None
else:
    model, actions = load_models_and_labels(MODEL_FILE, IS_DEEP_LEARNING)

# ==========================================
# MAIN APPLICATION UI
# ==========================================

# Header
st.markdown("""
<div class='main-header'>
    <h1 class='main-title'>American Sign Language Recognition</h1>
    <p class='sub-title'>Real-time gesture translation using computer vision and machine learning</p>
</div>
""", unsafe_allow_html=True)

# Check if model loaded successfully
if model is None:
    st.error(f"""
    ### Model Loading Error
    
    The model file was not found or could not be loaded.
    
    **Required files:**
    1. {MODEL_FILE} - Model weights
    2. {LABEL_MAP_FILE} - Label mapping
    
    **For Mini-XGBoost model only:**
    - {MODEL_FILES['minixgb']} - 5-word vocabulary
    
    **For full models (Bi-LSTM/XGBoost):**
    - {MODEL_FILES['bilstm']} or {MODEL_FILES['xgboost']}
    - {LABEL_MAP_FILE} - 14-word vocabulary mapping
    """)
    st.stop()

# Create main layout
col1, col2 = st.columns([0.65, 0.35])

with col1:
    st.markdown("### Live Camera Feed")
    
    # Camera controls
    col_start, col_stop = st.columns(2)
    with col_start:
        run_camera = st.button("Start Recognition", use_container_width=True)
    with col_stop:
        if st.button("Stop System", use_container_width=True):
            run_camera = False
            st.rerun()
    
    # Camera feed placeholder
    camera_container = st.empty()

with col2:
    st.markdown("### Recognition Results")
    
    # Results container
    result_container = st.empty()
    
    # System status
    st.markdown("#### System Status")
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.markdown("""
        <div class='status-indicator status-inactive' id='camera-status'>
            Camera: Offline
        </div>
        """, unsafe_allow_html=True)
    
    with status_col2:
        st.markdown("""
        <div class='status-indicator status-inactive' id='model-status'>
            Model: Idle
        </div>
        """, unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("#### Performance Metrics")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-value' id='fps-counter'>0</div>
            <div class='metric-label'>FPS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-value' id='confidence-display'>0%</div>
            <div class='metric-label'>Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-value' id='buffer-fill'>0%</div>
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
    st.markdown("""
    <script>
        document.getElementById('camera-status').className = 'status-indicator status-active';
        document.getElementById('camera-status').innerHTML = 'Camera: Active';
        document.getElementById('model-status').className = 'status-indicator status-active';
        document.getElementById('model-status').innerHTML = 'Model: Ready';
    </script>
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
                st.markdown(f"""
                <script>
                    document.getElementById('fps-counter').innerHTML = '{int(avg_fps)}';
                </script>
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
            if draw_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=2)
                )
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=3)
                )
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=3)
                )
            
            # Extract features
            keypoints = extract_features(results)
            sequence.append(keypoints)
            
            # Update buffer display
            buffer_fill = min(100, int((len(sequence) / SEQUENCE_LENGTH) * 100))
            st.markdown(f"""
            <script>
                document.getElementById('buffer-fill').innerHTML = '{buffer_fill}%';
            </script>
            """, unsafe_allow_html=True)
            
            # Perform inference when buffer is full
            current_prediction = None
            current_confidence = 0
            
            if len(sequence) == SEQUENCE_LENGTH:
                try:
                    if IS_DEEP_LEARNING and TF_AVAILABLE:
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
                        st.markdown(f"""
                        <script>
                            document.getElementById('confidence-display').innerHTML = '{int(current_confidence*100)}%';
                        </script>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.warning(f"Inference error: {str(e)[:50]}")
            
            # Display prediction
            if current_prediction:
                result_container.markdown(f"""
                <div class='prediction-card'>
                    <h3 style='color: #4a5568; margin: 0;'>DETECTED SIGN</h3>
                    <div class='prediction-word'>{current_prediction.upper()}</div>
                    <div class='confidence-bar'>
                        <div class='confidence-fill' style='width: {int(current_confidence*100)}%'></div>
                    </div>
                    <p style='color: #718096; font-weight: 600;'>
                        Confidence: {int(current_confidence*100)}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add overlay to video frame
                cv2.rectangle(image, (0, 0), (640, 70), (0, 0, 0), -1)
                cv2.putText(
                    image, f"Sign: {current_prediction}", 
                    (20, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2
                )
                cv2.putText(
                    image, f"Confidence: {int(current_confidence*100)}%", 
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            else:
                # Show waiting state
                result_container.markdown(f"""
                <div class='prediction-card'>
                    <h3 style='color: #4a5568; margin: 0;'>ANALYZING GESTURE</h3>
                    <div style='font-size: 3rem; color: #a0aec0; margin: 2rem;'>...</div>
                    <p style='color: #718096;'>Processing {len(sequence)}/{SEQUENCE_LENGTH} frames</p>
                    <div class='confidence-bar'>
                        <div class='confidence-fill' style='width: {buffer_fill}%'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add FPS counter to frame if enabled
            if show_fps and avg_fps > 0:
                cv2.putText(
                    image, f"FPS: {int(avg_fps)}", 
                    (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
            
            # Display frame
            frame_placeholder.image(image, channels="BGR")
    
    # Cleanup
    cap.release()
    
    # Final status update
    st.markdown("""
    <script>
        document.getElementById('camera-status').className = 'status-indicator status-inactive';
        document.getElementById('camera-status').innerHTML = 'Camera: Offline';
        document.getElementById('model-status').className = 'status-indicator status-inactive';
        document.getElementById('model-status').innerHTML = 'Model: Idle';
    </script>
    """, unsafe_allow_html=True)
    
    # Show session summary
    if prediction_history:
        st.markdown("---")
        st.markdown("### Session Summary")
        cols = st.columns(len(prediction_history))
        for idx, (pred, conf) in enumerate(prediction_history):
            with cols[idx]:
                st.markdown(f"""
                <div class='metric-box'>
                    <h4>{pred.upper()}</h4>
                    <p style='color: #718096; font-size: 0.9rem;'>
                        {int(conf*100)}% confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
else:
    # Welcome state
    result_container.markdown("""
    <div class='prediction-card'>
        <h2 style='color: #4a5568;'>Ready for Recognition</h2>
        <p style='color: #718096; margin: 1rem 0;'>
            Click <strong>Start Recognition</strong> to begin real-time ASL translation
        </p>
        <div style='font-size: 4rem; color: #c3cfe2; margin: 1rem;'>🖐️</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    ### System Requirements
    - Python 3.8+
    - Webcam (720p recommended)
    - 4GB RAM minimum
    """)

with footer_col2:
    st.markdown("""
    ### Supported Models
    - **Bi-LSTM**: 14 signs
    - **XGBoost**: 14 signs  
    - **Mini-XGBoost**: 5 signs
    """)

with footer_col3:
    st.markdown("""
    ### Tips
    - Ensure good lighting
    - Keep hands visible
    - Maintain consistent speed
    - Use plain background
    """)

st.caption(f"ASL Recognition System v1.0 | Model: {model_type} | Sequence Length: {SEQUENCE_LENGTH}")