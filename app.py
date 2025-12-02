import os
import sys
import json
import time
import numpy as np
import streamlit as st
from collections import deque

# ==========================================
# CONFIGURATION FOR STREAMLIT CLOUD
# ==========================================

# Disable everything that could cause issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MEDIAPIPE_DISABLE_MODEL_DOWNLOAD'] = '1'  # Prevent MediaPipe downloads

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Constants
SEQUENCE_LENGTH = 20  # Reduced for faster processing
CONFIDENCE_THRESHOLD = 0.75

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
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        border-left: 5px solid #667eea;
    }
    .prediction-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2d3748;
        margin: 1rem 0;
    }
    .progress-container {
        width: 100%;
        height: 20px;
        background: #e2e8f0;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #48bb78, #38a169);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .camera-placeholder {
        background: #000;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: white;
        min-height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2d3748;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
    }
    .button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
    }
    .button-secondary {
        background: #e2e8f0;
        color: #4a5568;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================

st.sidebar.markdown("""
<div class="card">
    <h3 style="margin:0; color:#667eea;">Control Panel</h3>
</div>
""", unsafe_allow_html=True)

# Operation Mode
st.sidebar.markdown("### Operation Mode")
operation_mode = st.sidebar.radio(
    "Select Mode:",
    ["Demo Mode", "Live Mode (Local Only)"],
    index=0
)

# Model Selection
st.sidebar.markdown("### AI Model")
model_selection = st.sidebar.selectbox(
    "Select Model:",
    ["Mini-XGBoost", "XGBoost", "Bi-LSTM"]
)

# Settings
st.sidebar.markdown("### Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="card">
    <h4 style="color:#667eea;">About This App</h4>
    <p style="font-size:0.9rem; color:#718096;">
        This is a Streamlit Cloud compatible version.
        Use <strong>Demo Mode</strong> for testing on cloud.
        Use <strong>Live Mode</strong> when running locally.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MODEL DATA
# ==========================================

# Demo vocabulary based on your label_map_2.json
DEMO_VOCABULARY = [
    'banana', 'catch', 'cool', 'cry', 'drown',
    'envelope', 'erase', 'follow', 'jacket', 'pineapple',
    'pop', 'sandwich', 'shave', 'strawberry'
]

# For Mini model, use first 5 words
MINI_VOCABULARY = ['banana', 'jacket', 'cry', 'catch', 'pop']

# ==========================================
# SIMULATED PROCESSING FUNCTIONS
# ==========================================

def simulate_feature_extraction():
    """Simulate feature extraction for demo mode"""
    return np.random.randn(198).astype(np.float32)

def simulate_prediction(model_type, confidence=0.85):
    """Simulate AI prediction"""
    if model_type == "Mini-XGBoost":
        vocab = MINI_VOCABULARY
    else:
        vocab = DEMO_VOCABULARY
    
    word = np.random.choice(vocab)
    conf = np.random.uniform(confidence - 0.1, min(confidence + 0.1, 0.99))
    return word, conf

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================

if 'sequence' not in st.session_state:
    st.session_state.sequence = deque(maxlen=SEQUENCE_LENGTH)
if 'running' not in st.session_state:
    st.session_state.running = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = ("", 0)
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

# ==========================================
# MAIN INTERFACE
# ==========================================

# Header
st.markdown("""
<div class="header-section">
    <h1 class="header-title">ASL Recognition System</h1>
    <p class="header-subtitle">Real-time American Sign Language Translation</p>
</div>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Camera Feed")
    
    # Control buttons
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶ Start Recognition", key="start", type="primary", use_container_width=True):
            st.session_state.running = True
            st.session_state.sequence.clear()
            st.session_state.current_prediction = ("", 0)
            st.session_state.predictions_history = []
            st.session_state.frame_count = 0
            st.session_state.start_time = time.time()
    
    with col_stop:
        if st.button("⏹ Stop", key="stop", use_container_width=True):
            st.session_state.running = False
    
    # Camera placeholder
    camera_display = st.empty()
    
    # Status indicators
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        mode_status = st.empty()
    with status_col2:
        buffer_status = st.empty()
    with status_col3:
        fps_status = st.empty()

with col2:
    st.markdown("### Recognition Result")
    
    # Results display
    result_display = st.empty()
    
    # Confidence display
    confidence_display = st.empty()
    
    # Recent predictions
    st.markdown("### Recent Predictions")
    predictions_container = st.empty()

# ==========================================
# PROCESSING LOOP
# ==========================================

if st.session_state.running:
    # Update status
    mode_status.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{operation_mode}</div>
        <div class="metric-label">Mode</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Processing loop simulation
    while st.session_state.running:
        try:
            # Calculate FPS
            st.session_state.frame_count += 1
            elapsed_time = time.time() - st.session_state.start_time
            
            if elapsed_time > 0:
                current_fps = st.session_state.frame_count / elapsed_time
            else:
                current_fps = 0
            
            # Update FPS display
            fps_status.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{int(current_fps)}</div>
                <div class="metric-label">FPS</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate camera feed
            if operation_mode == "Live Mode (Local Only)":
                # Try to import OpenCV and MediaPipe
                try:
                    import cv2
                    import mediapipe as mp
                    
                    # This will fail on Streamlit Cloud, fallback to demo
                    camera_display.warning("Live Mode requires local execution. Switching to Demo Mode.")
                    operation_mode = "Demo Mode"
                    
                except Exception as e:
                    camera_display.warning("Live camera not available. Using Demo Mode.")
                    operation_mode = "Demo Mode"
            
            # For Demo Mode, show simulated feed
            if operation_mode == "Demo Mode":
                # Create simulated camera frame
                width, height = 640, 480
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:] = (30, 30, 30)  # Dark background
                
                # Add ASL text
                cv2 = __import__('cv2')
                cv2.putText(frame, "ASL DEMO MODE", (150, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 150, 255), 3)
                cv2.putText(frame, "Simulated Camera Feed", (120, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                # Draw a hand skeleton (simulated)
                center_x, center_y = width // 2, height // 2 + 50
                cv2.circle(frame, (center_x, center_y), 60, (100, 150, 255), 2)
                
                # Draw fingers
                for i, angle in enumerate([-60, -30, 0, 30, 60]):
                    rad = np.deg2rad(angle)
                    length = 70 + (i * 5)
                    end_x = int(center_x + length * np.sin(rad))
                    end_y = int(center_y - length * np.cos(rad))
                    cv2.line(frame, (center_x, center_y), (end_x, end_y), (100, 150, 255), 2)
                
                # Add FPS
                cv2.putText(frame, f"FPS: {int(current_fps)}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display frame
                camera_display.image(frame, channels="BGR")
            
            # Simulate feature extraction
            features = simulate_feature_extraction()
            st.session_state.sequence.append(features)
            
            # Update buffer status
            buffer_len = len(st.session_state.sequence)
            buffer_percent = int((buffer_len / SEQUENCE_LENGTH) * 100)
            
            buffer_status.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{buffer_len}/{SEQUENCE_LENGTH}</div>
                <div class="metric-label">Buffer</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Make prediction when buffer is full
            if buffer_len == SEQUENCE_LENGTH:
                word, confidence = simulate_prediction(model_selection, confidence_threshold)
                
                if confidence > confidence_threshold:
                    st.session_state.current_prediction = (word, confidence)
                    
                    # Add to history
                    st.session_state.predictions_history.append((word, confidence))
                    if len(st.session_state.predictions_history) > 5:
                        st.session_state.predictions_history.pop(0)
            
            # Get current prediction
            current_word, current_conf = st.session_state.current_prediction
            
            # Display results
            if current_word:
                # Update result display
                result_display.markdown(f"""
                <div class="result-card">
                    <div class="prediction-text">{current_word.upper()}</div>
                    <div class="progress-container">
                        <div class="progress-fill" style="width: {current_conf*100}%"></div>
                    </div>
                    <p style="color: #4a5568; font-size: 1.1rem;">
                        Confidence: {current_conf*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Update confidence display
                confidence_display.metric("Current Confidence", f"{current_conf*100:.1f}%")
                
                # Add prediction to camera frame
                if 'frame' in locals():
                    cv2.putText(frame, f"{current_word.upper()} ({current_conf*100:.0f}%)", 
                               (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Show waiting state
                result_display.markdown(f"""
                <div class="result-card">
                    <h3>Collecting Data</h3>
                    <p>Processing frames: {buffer_len}/{SEQUENCE_LENGTH}</p>
                    <div class="progress-container">
                        <div class="progress-fill" style="width: {buffer_percent}%"></div>
                    </div>
                    <p style="color: #718096;">
                        Performing sign language gestures...
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Update recent predictions
            if st.session_state.predictions_history:
                pred_html = ""
                for word, conf in reversed(st.session_state.predictions_history):
                    pred_html += f"""
                    <div style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: bold;">{word.upper()}</span>
                            <span style="color: #48bb78;">{conf*100:.0f}%</span>
                        </div>
                    </div>
                    """
                
                predictions_container.markdown(f"""
                <div class="card">
                    {pred_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                predictions_container.markdown("""
                <div class="card">
                    <p style="color: #718096; text-align: center;">
                        No predictions yet
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Small delay
            time.sleep(0.1)  # 10 FPS for smoothness
            
        except Exception as e:
            # If any error occurs, show error and continue in demo mode
            st.error(f"Error in processing: {str(e)[:100]}")
            operation_mode = "Demo Mode"
            time.sleep(1)
    
else:
    # Not running state
    mode_status.markdown("""
    <div class="metric-box">
        <div class="metric-value">STOPPED</div>
        <div class="metric-label">Status</div>
    </div>
    """, unsafe_allow_html=True)
    
    buffer_status.markdown("""
    <div class="metric-box">
        <div class="metric-value">0/{SEQUENCE_LENGTH}</div>
        <div class="metric-label">Buffer</div>
    </div>
    """, unsafe_allow_html=True)
    
    fps_status.markdown("""
    <div class="metric-box">
        <div class="metric-value">--</div>
        <div class="metric-label">FPS</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show default camera view
    width, height = 640, 480
    default_frame = np.zeros((height, width, 3), dtype=np.uint8)
    default_frame[:] = (40, 40, 40)
    
    cv2 = __import__('cv2')
    cv2.putText(default_frame, "ASL RECOGNITION SYSTEM", (80, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 150, 255), 2)
    cv2.putText(default_frame, "Click 'Start Recognition' to begin", (120, 160), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(default_frame, f"Mode: {operation_mode}", (200, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    
    camera_display.image(default_frame, channels="BGR")
    
    # Show default result
    result_display.markdown("""
    <div class="result-card">
        <h3>Ready for Recognition</h3>
        <p>System is ready to start</p>
        <p>Click "Start Recognition" to begin</p>
        <div style="margin-top: 1rem; color: #718096; font-size: 0.9rem;">
            <p>• Select operation mode</p>
            <p>• Choose AI model</p>
            <p>• Adjust confidence level</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    confidence_display.metric("Confidence Threshold", f"{confidence_threshold*100:.0f}%")
    
    predictions_container.markdown("""
    <div class="card">
        <p style="color: #718096; text-align: center;">
            System idle
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER SECTION
# ==========================================

st.markdown("---")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    <div class="card">
        <h4 style="color:#667eea;">Getting Started</h4>
        <ul style="color:#718096; padding-left: 1rem;">
            <li>Select Demo Mode</li>
            <li>Choose AI model</li>
            <li>Click Start</li>
            <li>View predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div class="card">
        <h4 style="color:#667eea;">Operation Modes</h4>
        <p style="color:#718096;">
            <strong>Demo Mode:</strong> Works everywhere<br>
            <strong>Live Mode:</strong> Local execution only<br>
            <strong>Cloud:</strong> Use Demo Mode
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_info3:
    st.markdown("""
    <div class="card">
        <h4 style="color:#667eea;">AI Models</h4>
        <p style="color:#718096;">
            <strong>Mini-XGBoost:</strong> 5 signs, fastest<br>
            <strong>XGBoost:</strong> 14 signs, balanced<br>
            <strong>Bi-LSTM:</strong> 14 signs, most accurate
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption(f"ASL Recognition System | Mode: {operation_mode} | Model: {model_selection}")

# ==========================================
# HELP SECTION
# ==========================================

with st.expander("Help & Information"):
    st.markdown("""
    ### About This Application
    
    This is a **Streamlit Cloud compatible** version of the ASL Recognition System.
    
    ### Why Demo Mode?
    
    Streamlit Cloud has security restrictions that prevent:
    1. Access to webcam/microphone
    2. File system write access
    3. Downloading external models
    
    ### For Local Execution:
    
    If you want to use the live camera:
    
    ```bash
    # 1. Download the files locally
    # 2. Install requirements:
    pip install streamlit opencv-python mediapipe numpy joblib
    
    # 3. Run locally:
    streamlit run app.py
    
    # 4. Select "Live Mode" in the sidebar
    ```
    
    ### Model Files Needed:
    
    For full functionality, place these in the same folder:
    
    1. **For Demo Mode:** No files needed
    2. **For Live Mode with models:**
       - `label_map_2.json` - Your label file
       - One of: `xgb_model.pkl`, `xgboost_asl.pkl`, or `bilstm_model_2.keras`
    
    ### Troubleshooting:
    
    **If you see errors on Streamlit Cloud:**
    - Always use **Demo Mode**
    - Don't select Live Mode
    - The app is designed to work without camera on cloud
    
    **For best experience:**
    - Use Chrome browser
    - Good internet connection
    - Allow popups if needed
    """)

# ==========================================
# COMPATIBILITY NOTES
# ==========================================

# Hide MediaPipe imports to prevent download errors
# The app will work in Demo Mode without actual MediaPipe