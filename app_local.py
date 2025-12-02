import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import json
import time
from collections import deque

# ==========================================
# 1. Page Configuration & Professional Custom CSS
# ==========================================
st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Professional CSS
st.markdown("""
    <style>
    /* Main Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .main-title {
        font-size: 3rem;
        color: white;
        font-weight: 800;
        letter-spacing: 1px;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-title {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Card Styling */
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    
    /* Prediction Display */
    .prediction-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        border-left: 6px solid #667eea;
        min-height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .prediction-word {
        font-size: 2.8rem;
        font-weight: 800;
        color: #2d3748;
        margin: 1rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
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
    
    .confidence-text {
        font-size: 1.1rem;
        color: #4a5568;
        font-weight: 600;
    }
    
    /* Status Indicators */
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: 600;
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
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    .model-info {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Camera Feed Container */
    .camera-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        background: #000;
        position: relative;
    }
    
    /* Performance Metrics */
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-top: 4px solid #667eea;
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
        letter-spacing: 1px;
    }
    
    /* Fix for Streamlit deprecated warning */
    .stImage > img {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Configuration & Constants
# ==========================================
SEQUENCE_LENGTH = 50
MINI_CLASSES = ['banana', 'jacket', 'cry', 'catch', 'pop']

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 3. Sidebar: Control Panel
# ==========================================
st.sidebar.markdown("""
    <div class='card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center;'>
        <h3 style='margin: 0;'>ASL Control Panel</h3>
    </div>
""", unsafe_allow_html=True)

# --- Model Selection ---
st.sidebar.markdown("### Model Architecture")
model_type = st.sidebar.selectbox(
    "Select Model Type:",
    ("Bi-LSTM (Deep Learning - 14 Words)", 
     "XGBoost (Machine Learning - 14 Words)",
     "Mini-XGBoost (Experimental - 5 Words)")
)

# --- Parameters ---
st.sidebar.markdown("### Configuration Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.75, 0.05)
draw_landmarks = st.sidebar.checkbox("Show Skeleton Overlay", value=True)
show_fps = st.sidebar.checkbox("Display FPS Counter", value=True)

# --- Model Info Display ---
st.sidebar.markdown("### Model Specifications")

if "Bi-LSTM" in model_type:
    MODEL_FILE = 'bilstm_model_2.keras'
    IS_DEEP_LEARNING = True
    st.sidebar.markdown("""
        <div class='model-info'>
            <h4>Bi-LSTM Neural Network</h4>
            <p><strong>Architecture:</strong> Bidirectional LSTM with Hybrid Pooling</p>
            <p><strong>Input:</strong> Raw Time-Series (50 frames)</p>
            <p><strong>Accuracy:</strong> 87.86%</p>
            <p><strong>Best For:</strong> Complex temporal patterns</p>
        </div>
    """, unsafe_allow_html=True)
    
elif "XGBoost" in model_type and not "Mini" in model_type:
    MODEL_FILE = 'xgboost_asl.pkl'
    IS_DEEP_LEARNING = False
    st.sidebar.markdown("""
        <div class='model-info'>
            <h4>XGBoost Classifier</h4>
            <p><strong>Architecture:</strong> Gradient Boosting Decision Trees</p>
            <p><strong>Input:</strong> Statistical Features (792 features)</p>
            <p><strong>Accuracy:</strong> 79.56%</p>
            <p><strong>Best For:</strong> Fast inference, distinct poses</p>
        </div>
    """, unsafe_allow_html=True)

else:
    MODEL_FILE = 'xgb_model.pkl'
    IS_DEEP_LEARNING = False
    st.sidebar.markdown("""
        <div class='model-info'>
            <h4>Mini XGBoost (Demo)</h4>
            <p><strong>Scope:</strong> 5 basic signs</p>
            <p><strong>Purpose:</strong> Demonstration & rapid testing</p>
            <p><strong>Vocabulary:</strong> banana, jacket, cry, catch, pop</p>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. Preprocessing Engine
# ==========================================
def normalize_hand(pts):
    """Normalize hand landmarks relative to wrist"""
    ref = pts[0].copy()
    scale = np.linalg.norm(pts[9] - ref)
    if scale < 1e-6: scale = 1.0
    return (pts - ref) / scale

def compute_torso_stats(pose_landmarks):
    """Compute torso center and scale for normalization"""
    torso_center = np.array([0.5, 0.5], dtype=np.float32)
    torso_scale = 1.0
    try:
        def get_xy(idx): 
            return np.array([pose_landmarks.landmark[idx].x, 
                           pose_landmarks.landmark[idx].y], dtype=np.float32)
        
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
    
    # Pose landmarks (0-65)
    torso_center = np.array([0.5, 0.5], dtype=np.float32)
    torso_scale = 1.0
    if results.pose_landmarks:
        torso_center, torso_scale = compute_torso_stats(results.pose_landmarks)
        pose_xy = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark], 
                          dtype=np.float32)
        pose_norm = (pose_xy - torso_center[None, :]) / torso_scale
        feat[0:66] = pose_norm.flatten()

    # Left hand (66-131)
    if results.left_hand_landmarks:
        l_pts = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], 
                        dtype=np.float32)
        feat[66:129] = normalize_hand(l_pts)[:, :3].flatten()
        wrist_rel = l_pts[0].copy()
        wrist_rel[:2] = (wrist_rel[:2] - torso_center) / torso_scale
        wrist_rel[2] /= torso_scale
        feat[129:132] = wrist_rel

    # Right hand (132-197)
    if results.right_hand_landmarks:
        r_pts = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], 
                        dtype=np.float32)
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
# 5. Resource Loading (Cached)
# ==========================================
@st.cache_resource
def load_models_and_labels(model_path, is_dl):
    """Load model and labels with caching"""
    model = None
    actions = {}
    
    try:
        # Load model
        if is_dl:
            model = tf.keras.models.load_model(model_path)
        else:
            model = joblib.load(model_path)
            
        # Load labels
        if "Mini" in model_type:
            actions = {i: label for i, label in enumerate(MINI_CLASSES)}
        else:
            with open('label_map_2.json', 'r') as f:
                label_map = json.load(f)
            if isinstance(label_map, dict):
                actions = {v: k for k, v in label_map.items()}
            else:
                actions = {i: label for i, label in enumerate(label_map)}
                
        return model, actions
        
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

# Load resources
model, actions = load_models_and_labels(MODEL_FILE, IS_DEEP_LEARNING)

# ==========================================
# 6. Main UI Layout
# ==========================================
st.markdown("""
    <div class='main-header'>
        <h1 class='main-title'>American Sign Language Recognition</h1>
        <p class='sub-title'>Real-time gesture translation using advanced computer vision and machine learning</p>
    </div>
""", unsafe_allow_html=True)

# Check if model loaded successfully
if model is None:
    st.error(f"""
    ### Model Loading Error
    The model file `{MODEL_FILE}` was not found in the application directory.
    
    **Please ensure:**
    1. The model file exists in the same folder as this application
    2. The filename matches exactly: `{MODEL_FILE}`
    3. For full models, ensure `label_map_2.json` is also present
    """)
    
    with st.expander("Troubleshooting Guide"):
        st.markdown("""
        ### Common Solutions:
        
        1. **Place model files in the app directory:**
           - `bilstm_model_2.keras` for Bi-LSTM
           - `xgboost_asl.pkl` for XGBoost
           - `xgb_model.pkl` for Mini-XGBoost
           - `label_map_2.json` for label mapping
        
        2. **Check file permissions**
        3. **Verify TensorFlow/joblib versions match training environment**
        """)
    st.stop()

# Create main layout columns
col1, col2 = st.columns([0.65, 0.35])

with col1:
    st.markdown("### Live Camera Feed")
    
    # Camera controls
    col_start, col_stop = st.columns(2)
    with col_start:
        run_camera = st.button("Start Recognition System", 
                             help="Begin real-time ASL recognition",
                             use_container_width=True)
    with col_stop:
        stop_button = st.button("Stop System", use_container_width=True)
        if stop_button:
            run_camera = False
            st.rerun()
    
    # Camera feed container
    camera_placeholder = st.empty()
    with camera_placeholder.container():
        st.markdown('<div class="camera-container">', unsafe_allow_html=True)
        frame_window = st.image([], use_container_width=True)  # تم التصحيح هنا
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### Recognition Results")
    
    # Result display container
    result_container = st.empty()
    
    # System status
    st.markdown("#### System Status")
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.markdown("""
            <div class='status-box status-inactive' id='camera-status'>
                Camera: Offline
            </div>
        """, unsafe_allow_html=True)
    
    with status_col2:
        st.markdown("""
            <div class='status-box status-inactive' id='model-status'>
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
# 7. Recognition Loop
# ==========================================
sequence = deque(maxlen=SEQUENCE_LENGTH)
last_prediction = None
prediction_history = []
fps_buffer = deque(maxlen=30)
avg_fps = 0  # تهيئة المتغير قبل الاستخدام

if run_camera:
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Status update
    st.markdown("""
        <script>
            document.getElementById('camera-status').className = 'status-box status-active';
            document.getElementById('camera-status').innerHTML = 'Camera: Active';
            document.getElementById('model-status').className = 'status-box status-active';
            document.getElementById('model-status').innerHTML = 'Model: Running';
        </script>
    """, unsafe_allow_html=True)
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time > 0:
                current_fps = frame_count / elapsed_time
                fps_buffer.append(current_fps)
                
                # تحديث avg_fps قبل استخدامه
                if fps_buffer:
                    avg_fps = np.mean(fps_buffer)
                else:
                    avg_fps = 0
                
                # Update FPS display
                st.markdown(f"""
                    <script>
                        document.getElementById('fps-counter').innerHTML = '{int(avg_fps)}';
                    </script>
                """, unsafe_allow_html=True)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                st.warning("Unable to access camera. Please check connection.")
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
                # Update model status
                st.markdown("""
                    <script>
                        document.getElementById('model-status').innerHTML = 'Model: Processing';
                    </script>
                """, unsafe_allow_html=True)
                
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
                    last_prediction = current_prediction
                    
                    # Add to history (keep last 5)
                    prediction_history.append((current_prediction, current_confidence))
                    if len(prediction_history) > 5:
                        prediction_history.pop(0)
                    
                    # Update confidence display
                    st.markdown(f"""
                        <script>
                            document.getElementById('confidence-display').innerHTML = '{int(current_confidence*100)}%';
                        </script>
                    """, unsafe_allow_html=True)
            
            # Display prediction
            if current_prediction:
                result_container.markdown(f"""
                    <div class='prediction-container'>
                        <h3 style='color: #4a5568; margin: 0;'>DETECTED SIGN</h3>
                        <div class='prediction-word'>{current_prediction.upper()}</div>
                        <div class='confidence-bar'>
                            <div class='confidence-fill' style='width: {int(current_confidence*100)}%'></div>
                        </div>
                        <p class='confidence-text'>Confidence: {int(current_confidence*100)}%</p>
                        <p style='color: #718096; font-size: 0.9rem; margin-top: 1rem;'>
                            Model: {model_type.split(' ')[0]} | Threshold: {conf_threshold}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add overlay to video frame
                cv2.rectangle(image, (0, 0), (640, 70), (0, 0, 0), -1)
                cv2.putText(image, f"Sign: {current_prediction}", 
                           (20, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f"Confidence: {int(current_confidence*100)}%", 
                           (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Show waiting state
                result_container.markdown(f"""
                    <div class='prediction-container'>
                        <h3 style='color: #4a5568; margin: 0;'>ANALYZING GESTURE</h3>
                        <div style='font-size: 3rem; color: #a0aec0; margin: 2rem;'>...</div>
                        <p style='color: #718096;'>Processing {len(sequence)}/{SEQUENCE_LENGTH} frames</p>
                        <div class='confidence-bar'>
                            <div class='confidence-fill' style='width: {buffer_fill}%'></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Add FPS counter to frame if enabled - التصحيح هنا
            if show_fps and avg_fps > 0:
                cv2.putText(image, f"FPS: {int(avg_fps)}", 
                           (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            frame_window.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final status update
    st.markdown("""
        <script>
            document.getElementById('camera-status').className = 'status-box status-inactive';
            document.getElementById('camera-status').innerHTML = 'Camera: Offline';
            document.getElementById('model-status').className = 'status-box status-inactive';
            document.getElementById('model-status').innerHTML = 'Model: Idle';
        </script>
    """, unsafe_allow_html=True)
    
    # Show session summary
    if prediction_history:
        st.markdown("---")
        st.markdown("### Session Summary")
        summary_cols = st.columns(len(prediction_history))
        for idx, (pred, conf) in enumerate(prediction_history):
            with summary_cols[idx]:
                st.markdown(f"""
                    <div class='card' style='text-align: center;'>
                        <h4>{pred.upper()}</h4>
                        <div style='font-size: 0.8rem; color: #718096;'>{int(conf*100)}% confidence</div>
                    </div>
                """, unsafe_allow_html=True)
else:
    # Welcome state
    result_container.markdown("""
        <div class='prediction-container'>
            <h2 style='color: #4a5568;'>Ready for Recognition</h2>
            <p style='color: #718096; margin: 1rem 0;'>
                Click <strong>Start Recognition System</strong> to begin real-time ASL translation
            </p>
            <div style='font-size: 4rem; color: #c3cfe2; margin: 1rem;'>🖐️</div>
            <p style='color: #718096; font-size: 0.9rem;'>
                Ensure proper lighting and clear view of your hands
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show instructions
    with st.expander("Getting Started Instructions", expanded=True):
        col_inst1, col_inst2, col_inst3 = st.columns(3)
        
        with col_inst1:
            st.markdown("""
                ### Camera Setup
                1. Grant camera permissions
                2. Ensure good lighting
                3. Position camera at chest level
                4. Maintain clear background
            """)
        
        with col_inst2:
            st.markdown("""
                ### Signing Tips
                1. Keep hands within frame
                2. Maintain consistent pace
                3. Hold signs for 1-2 seconds
                4. Avoid rapid movements
            """)
        
        with col_inst3:
            st.markdown("""
                ### Best Results
                1. Start with Mini model
                2. Adjust confidence threshold
                3. Enable skeleton overlay
                4. Review performance metrics
            """)

# ==========================================
# 8. Footer & Additional Information
# ==========================================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
        ### System Requirements
        - Python 3.8+
        - Webcam (720p+ recommended)
        - 4GB RAM minimum
        - Modern browser
    """)

with footer_col2:
    st.markdown("""
        ### Supported Models
        - **Bi-LSTM**: Complex sequences
        - **XGBoost**: Fast inference
        - **Mini**: Quick testing
    """)

with footer_col3:
    st.markdown("""
        ### Tips for Accuracy
        1. Use consistent signing speed
        2. Maintain proper distance
        3. Ensure good contrast
        4. Calibrate in neutral pose
    """)

# System info
st.markdown("---")
st.caption(f"ASL Recognition System | Model: {model_type} | Buffer Size: {SEQUENCE_LENGTH} frames")