import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import mediapipe as mp

# ==============================================================================
# 1. Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="RPG Motion Game",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 RPG Motion Tracker")
st.write("Real-time stance and gesture tracking powered by MediaPipe and Streamlit WebRTC.")

# WebRTC ICE servers (Required for Cloud deployment connectivity)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==============================================================================
# 2. MediaPipe Pose & Drawing Initialization
# ==============================================================================
@st.cache_resource
def load_mediapipe_pose():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp.solutions.drawing_utils, mp_pose

pose, mp_drawing, mp_pose = load_mediapipe_pose()

# ==============================================================================
# 3. Colab Processing Logic (WebRTC Frame Callback)
# ==============================================================================
def process_colab_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Identical logic to Google Colab processing:
    Takes a BGR NumPy frame, processes MediaPipe pose landmarks,
    and draws the annotations directly on the frame.
    """
    # Flip camera horizontally for intuitive mirror mode
    frame_bgr = cv2.flip(frame_bgr, 1)

    # Convert BGR (OpenCV format) to RGB (MediaPipe requirement)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Perform MediaPipe detection
    results = pose.process(frame_rgb)

    # Draw pose landmark connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )

    return frame_bgr


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert incoming stream frame to OpenCV BGR array
    img_bgr = frame.to_ndarray(format="bgr24")

    # Run processing pipeline
    processed_bgr = process_colab_frame(img_bgr)

    # Return frame to WebRTC player
    return av.VideoFrame.from_ndarray(processed_bgr, format="bgr24")

# ==============================================================================
# 4. WebRTC Video Streamer Component
# ==============================================================================
webrtc_streamer(
    key="rpg-colab-streamer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
