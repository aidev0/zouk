from datetime import datetime

import cv2
import mediapipe as mp
from IPython.display import display, clear_output, Image
from PIL import Image as PILImage
from io import BytesIO
import warnings
import json
import time

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize MediaPipe pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils


# Function to capture frames from camera
def capture_frames_from_camera(num_frames=100):
    cap = cv2.VideoCapture(0)
    frames = []

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return frames

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


# Function to perform pose estimation on frames and save pose data
def perform_pose_estimation(frames, save_pose_data=False):
    processed_frames = []
    pose_data = []

    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            if save_pose_data:
                landmarks = [{
                    'x': landmark.x, 'y': landmark.y, 'z': landmark.z, 'visibility': landmark.visibility
                } for landmark in results.pose_landmarks.landmark]
                pose_data.append(landmarks)

        processed_frames.append(frame)

    if save_pose_data:
        with open('pose_data.json', 'w') as f:
            json.dump(pose_data, f, indent=4)

    return processed_frames


# Function to display frames in Jupyter notebook
def display_frames(frames):
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = PILImage.fromarray(rgb_frame)
        bio = BytesIO()
        pil_im.save(bio, format='JPEG')
        clear_output(wait=True)
        display(Image(data=bio.getvalue()))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Function to save frames as a video
def save_video(frames, output_path, fps=30):
    if len(frames) == 0:
        print("No frames to save.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


# Main workflow
print("Capturing frames from camera...")
time.sleep(5)
print("starting in 3 seconds...")
time.sleep(1)
print("starting in 2 seconds...")
time.sleep(1)
print("starting in 1 second...")
time.sleep(1)
print("starting now...")
num_frames_to_capture = 300
frames = capture_frames_from_camera(num_frames_to_capture)
processed_frames = perform_pose_estimation(frames, save_pose_data=True)
now = datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M-%S")
save_video(processed_frames, f'results/processed_video_{now}.mp4')
display_frames(processed_frames)
