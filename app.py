import streamlit as st
import numpy as np
import math, pickle
from PIL import Image
import cv2
import mediapipe as mp
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# Model loading
load_model = pickle.load(open('YogaModel.pkl', 'rb'))

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return round(ang + 360 if ang < 0 else ang)

def feature_list(poseLandmarks, posename):
    return [
        getAngle(poseLandmarks[16], poseLandmarks[14], poseLandmarks[12]),
        getAngle(poseLandmarks[14], poseLandmarks[12], poseLandmarks[24]),
        getAngle(poseLandmarks[13], poseLandmarks[11], poseLandmarks[23]),
        getAngle(poseLandmarks[15], poseLandmarks[13], poseLandmarks[11]),
        getAngle(poseLandmarks[12], poseLandmarks[24], poseLandmarks[26]),
        getAngle(poseLandmarks[11], poseLandmarks[23], poseLandmarks[25]),
        getAngle(poseLandmarks[24], poseLandmarks[26], poseLandmarks[28]),
        getAngle(poseLandmarks[23], poseLandmarks[25], poseLandmarks[27]),
        getAngle(poseLandmarks[26], poseLandmarks[28], poseLandmarks[32]),
        getAngle(poseLandmarks[25], poseLandmarks[27], poseLandmarks[31]),
        getAngle(poseLandmarks[0], poseLandmarks[12], poseLandmarks[11]),
        getAngle(poseLandmarks[0], poseLandmarks[11], poseLandmarks[12]),
        posename
    ]

# Streamlit layout
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.title('Yogibaba')
app_mode = st.sidebar.selectbox('Select The Pose', ['Tree', 'Mountain', 'Warrior2'])

# Display pose information and image
pose_info = {
    'Tree': ('Tree Pose', 'tree.jpg', 1),
    'Mountain': ('Mountain Pose', 'mountain.jpg', 2),
    'Warrior2': ('Warrior2 Pose', 'warrior2.jpg', 3)
}
pose_name, pose_image, posename = pose_info[app_mode]
st.title(f"{pose_name}")
col1, col2 = st.columns([2, 3])

with col1:
    image = Image.open(pose_image)
    st.image(image, caption=f'{pose_name}')

# VideoTransformer with MediaPipe Pose Detection
class PoseDetector(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        poseLandmarks = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                poseLandmarks.append((int(lm.x * w), int(lm.y * h)))

            if len(poseLandmarks) > 0:
                data = feature_list(poseLandmarks, posename)
                accuracy = int(round(load_model.predict(np.array(data).reshape(1, -1))[0], 0))
                cv2.putText(img, f"Accuracy: {accuracy}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        return img

# Set RTC Configuration with STUN servers for WebRTC
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# Webcam Stream with streamlit-webrtc
with col2:
    st.write("Webcam Live Feed")
    webrtc_streamer(
        key="pose-detection",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=PoseDetector,
        rtc_configuration=rtc_configuration
    )

