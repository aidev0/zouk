# Build
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Run
```bash
python3 pose_estimation/video.py
```

# Current Status
- It records the video from the camera.
- Extracts pose estimation from the video.
- Draws the pose estimation on the video.
- Saves to results directory.

# Next Steps
- Use joints features to analyze some movements like basic steps or head movements.
- Implement a user interface to play music, YouTube video, record video, analyze video, etc.
- Online Jack and Jill competition.