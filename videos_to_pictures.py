"""
Video to Images Converter
-------------------------
This script extracts frames from a video at a fixed interval, rotates them 90° clockwise,
crops them to a square centered on the image, and saves them as individual image files.
"""

import os
import cv2



# ------------------------------
# Configuration
# ------------------------------

video_path = ("C:/Users/Lukas/OneDrive - Kantonsschule Uetikon am See/"
    "Schule/Maturaarbeit/Videos_to_pictures/bruno.mp4")
output_folder = (
    "C:/Users/Lukas/OneDrive - Kantonsschule Uetikon am See/"
    "Schule/Maturaarbeit/Bruno"
)

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Frame extraction settings
frame_interval_sec = 0.2  # Extract one frame every 0.2 seconds
first_image_number = 1
first_image_name = f"bruno_{first_image_number}"



# ------------------------------
# Open video
# ------------------------------

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, int(fps * frame_interval_sec))

frame_count = 0
image_count = 0



# ------------------------------
# Process video frames
# ------------------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    if frame_count % frame_interval == 0:
        # Rotate frame 90° clockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Crop frame to square (centered)
        height, width, _ = frame.shape
        min_dim = min(height, width)
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        frame = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

        # Save frame
        if image_count == 0:
            frame_filename = f"{first_image_name}.jpg"
        else:
            frame_filename = f"bruno_{first_image_number + image_count}.jpg"

        cv2.imwrite(os.path.join(output_folder, frame_filename), frame)
        image_count += 1

    frame_count += 1



# ------------------------------
# Release resources
# ------------------------------

cap.release()
cv2.destroyAllWindows()

print(f"Total images saved: {image_count}")
