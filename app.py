import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np
from PIL import Image

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Streamlit app
st.set_page_config(page_title="Human Pose Estimation", layout="wide")
st.title("Human Pose Estimation Project")

# Function to process video for pose estimation
def process_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    with mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally and process it
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Draw landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

            # Display the processed frame
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    cap.release()

# Function to process real-time webcam feed
def process_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    stframe = st.empty()

    with mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally and process it
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Draw landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

            # Display the processed frame
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    cap.release()

# Function to process image for pose estimation
def process_image(image):
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, model_complexity=0, enable_segmentation=False,
                      min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)

        # Draw landmarks on the image
        if results.pose_landmarks:
            annotated_image = image_rgb.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            st.image(annotated_image, channels="RGB", use_column_width=True)
        else:
            st.warning("No pose landmarks detected in the image.")

# Streamlit sections
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Video Pose Estimation", "Real-Time Pose Estimation", "Image Pose Estimation"])

if section == "Video Pose Estimation":
    st.header("Pose Estimation on Video")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if video_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())
        st.text("Processing video...")
        process_video_stream(temp_file.name)

elif section == "Real-Time Pose Estimation":
    st.header("Pose Estimation with Webcam")
    st.text("Ensure your webcam is connected and click the button below to start.")
    if st.button("Start Webcam Pose Estimation"):
        process_webcam()

elif section == "Image Pose Estimation":
    st.header("Pose Estimation on Image")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.text("Processing image...")
        process_image(image)
