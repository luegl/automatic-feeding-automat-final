"""
Raspberry Pi Cat Image Capture
------------------------------
This program captures images from the Raspberry Pi camera and saves them
to a folder corresponding to the selected cat. Change the CAT_NAME variable
to either 'bruno' or 'flaekli' to capture images for the respective cat.
"""

# Standard library
import os
import time

# Third-party libraries
import cv2
from picamera2 import Picamera2



# -------------------------------------------------------------------
#   Configuration
# -------------------------------------------------------------------

CAT_NAME = "bruno"  
DATA_FOLDER = "data"
CAPTURE_INTERVAL = 0.5  # Time in seconds between captures
IMG_WIDTH, IMG_HEIGHT = 640, 480

# Ensure output folder exists
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, CAT_NAME)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)



# -------------------------------------------------------------------
#   Functions
# -------------------------------------------------------------------

def initialize_camera() -> Picamera2:
    """
    Initialize the Raspberry Pi camera.

    Returns:
        Picamera2: Configured and started camera object.
    """
    print("Initializing camera...")
    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"size": (IMG_WIDTH, IMG_HEIGHT)}
    )
    picam.configure(config)
    picam.start()
    print("Camera started, warming up for 2 seconds...")
    time.sleep(2)
    return picam


def capture_images(picam: Picamera2, folder: str, interval: float = 0.5) -> None:
    """
    Capture images from the camera at a fixed interval and save them to disk.

    Args:
        picam (Picamera2): Camera object.
        folder (str): Folder to save captured images.
        interval (float): Time between captures in seconds.
    """
    index = len(os.listdir(folder))  # Start counting from existing files

    try:
        while True:
            img = picam.capture_array()
            if img is None:
                print("No image captured, retrying...")
                time.sleep(0.1)
                continue

            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            filename = os.path.join(folder, f"{CAT_NAME}_{index}.jpg")
            cv2.imwrite(filename, img_bgr)
            print(f"Saved image: {filename}")

            index += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print("Image capture stopped by user.")
    finally:
        picam.stop()
        cv2.destroyAllWindows()



# -------------------------------------------------------------------
#   Main
# -------------------------------------------------------------------

def main() -> None:
    """
    Main function to initialize the camera and start capturing images.
    """
    picam = initialize_camera()
    print(
        f"Starting image capture for '{CAT_NAME}' in folder '{OUTPUT_FOLDER}'..."
    )
    capture_images(picam, OUTPUT_FOLDER, CAPTURE_INTERVAL)


if __name__ == "__main__":
    main()
