"""
Automatic Cat Feeder
-------------------
This program controls a smart feeding station with camera recognition
and weight sensor. It identifies cats via AI (YOLO + Keras),
opens the bowl for the correct cat, and monitors the weight to
control the ration.
"""

# Standard library
import json
import os
import time

# Third-party libraries
import cv2
import numpy as np
from PIL import Image
import RPi.GPIO as GPIO
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO
from picamera2 import Picamera2

# Local / custom libraries
from JoyIT_hx711py import HX711


# -------------------------------------------------------------------
#   Hardware Initialization (Scale, Motor, GPIO)
# -------------------------------------------------------------------

# HX711 scale initialization
HX = HX711(5, 6)
offset = float(input("Enter scale offset: "))
scale = float(input("Enter scale factor: "))
HX.set_offset(offset)
HX.set_scale(scale)

# Motor pins
DIR_PIN = 13
PUL_PIN = 11
ENA_PIN = 15

GPIO.setmode(GPIO.BOARD)
GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(PUL_PIN, GPIO.OUT)


# -------------------------------------------------------------------
#   Classes
# -------------------------------------------------------------------

class FoodBowl:
    """Represents a single food bowl."""

    def __init__(self, state: str, cat: str, weight: float):
        self.state = state
        self.cat = cat
        self.weight = weight



# -------------------------------------------------------------------
#   JSON Handling (Cat and Ration Data)
# -------------------------------------------------------------------

def load_cats(file_path: str) -> dict:
    """Load cat information from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


def save_cats(cats_data: dict, file_path: str = "cats.json") -> None:
    """Save updated cat information to a JSON file."""
    with open(file_path, "w") as file:
        json.dump(cats_data, file, indent=2)



# -------------------------------------------------------------------
#   Hardware Functions
# -------------------------------------------------------------------

def weigh_bowl() -> float:
    """Measure the bowl weight via the HX711 sensor."""
    HX.power_up()
    value = HX.get_grams()
    HX.power_down()
    return value


def step_motor(steps: int, direction: bool, delay: float = 0.001) -> None:
    """Control the stepper motor for a specific number of steps."""
    GPIO.output(DIR_PIN, direction)
    for _ in range(steps):
        GPIO.output(PUL_PIN, True)
        time.sleep(delay)
        GPIO.output(PUL_PIN, False)
        time.sleep(delay)


def open_bowl(cat_name: str, bowl: FoodBowl) -> None:
    """Open the bowl for the recognized cat."""
    step_motor(900, direction=True)
    bowl.state = "open"
    bowl.cat = cat_name


def close_bowl(weight: float, cat_name: str, bowl: FoodBowl, cats_data: dict) -> None:
    """Close the bowl and update remaining ration."""
    step_motor(900, direction=False)
    cats_data[cat_name]['ration_left'] = cats_data.get(cat_name, {}).get("ration_left")-(bowl.weight-weight)
    save_cats(cats_data)
    bowl.weight=weight
    bowl.state = "closed"
    bowl.cat = ""


def fill_up_bowl(bowl: FoodBowl, weight: float, cats_data: dict) -> None:
    """Automatically refill the bowl if the weight is too low."""
    if bowl.weight < max(cat_info["ration_total"] for cat_info in cats_data.values()):
        step_motor(287, direction=False)
        while int(weigh_bowl())<max(cat_info["ration_total"] for cat_info in cats_data.values()):
            time.sleep(0.001)
        step_motor(287, direction=True)
    bowl.weight = int(weigh_bowl())



# -------------------------------------------------------------------
#   AI Detection
# -------------------------------------------------------------------

def detect_cat(
    picam: Picamera2,
    yolo_model: YOLO,
    keras_models: dict,
    cat_names: list,
    img_size: tuple[int, int]
) -> str:
    """
    Detect if a cat is in front of the camera and identify it.

    Returns:
        cat_name (str): Name of the recognized cat or empty string.
    """
    img_array = picam.capture_array()
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    try:
        results = yolo_model.predict(source=img_array, conf=0.3, classes=[15])
    except Exception as err:
        print(f"YOLO detection error: {err}")
        return ""

    if not results or not results[0].boxes or len(results[0].boxes) == 0:
        return ""

    boxes = results[0].boxes.xyxy.tolist()
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cropped = img_array[y_min:y_max, x_min:x_max]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cropped.astype("uint8"))
        resized = image.resize(img_size)

        img_array_resized = keras.utils.img_to_array(resized) / 255.0
        img_array_resized = np.expand_dims(img_array_resized, axis=0)

        # Evaluate with individual cat models
        scores = []
        for cat in cat_names:
            try:
                score = float(keras_models[cat].predict(img_array_resized)[0])
                if 100 * (1 - score) > 25:
                    scores.append((cat, score))
            except Exception as e:
                print(f"Error with model '{cat}': {e}")

        if not scores:
            return ""
        return max(scores, key=lambda x: x[1])[0]

    return ""



# -------------------------------------------------------------------
#   Main Loop
# -------------------------------------------------------------------

def main_loop(
    picam: Picamera2,
    yolo_model: YOLO,
    keras_models: dict,
    cats_data: dict,
    bowl: FoodBowl,
    img_size: tuple[int, int]
) -> None:
    """Infinite loop for the cat feeder behavior."""
    wrong_detection_count = 0
    last_modified = os.path.getmtime("cats.json")

    while True:
        # Check if cat configuration changed
        current_modified = os.path.getmtime("cats.json")
        if current_modified != last_modified:
            last_modified = current_modified
            cats_data = load_cats("cats.json")

        cat_names = list(cats_data.keys())

        # Detect cat and measure weight
        cat = detect_cat(picam, yolo_model, keras_models, cat_names, img_size)
        weight = weigh_bowl()

        # Bowl open/close/refill logic
        if cat in cat_names and bowl.state == "closed" and int(cats_data[cat]["ration_left"]) > 0:
            open_bowl(cat, bowl)

        if bowl.state == "open":
            if cat != bowl.cat:
                wrong_detection_count += 1
                print(f"Wrong cat detected ({wrong_detection_count})")
            else:
                wrong_detection_count = 0

            if (bowl.weight - weight > cats_data.get(bowl.cat, {}).get("ration_left")) or (wrong_detection_count >= 5):
                close_bowl(weight, bowl.cat, bowl, cats_data)
                wrong_detection_count = 0
                fill_up_bowl(bowl, weight, cats_data)

        print(f"Current bowl state: {bowl.state}")
        time.sleep(0.1)



# -------------------------------------------------------------------
#   Program Start
# -------------------------------------------------------------------

def main():
    """Initialize camera, models, and start the feeder."""
    print("Starting automatic cat feeder...")

    # Load models
    MODELS_DIR = "models"
    keras_models = {}

    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".keras"):
            cat_name = filename.replace("modelv1_", "").replace(".keras", "")
            model_path = os.path.join(MODELS_DIR, filename)
            keras_models[cat_name] = tf.keras.models.load_model(model_path)
            print(f"Loaded model: {cat_name}")

    yolo_model = YOLO("yolov8n.pt")

    # Start camera
    picam = Picamera2()
    picam.start()
    print("Camera initialized")

    # Load JSON data
    cats_data = load_cats("cats.json")

    # Fill the bowl initially
    step_motor(287, direction=False)
    while int(weigh_bowl()) < max(cat_info["ration_total"] for cat_info in cats_data.values()):
        time.sleep(0.001)
    step_motor(287, direction=True)

    # Create bowl object
    bowl = FoodBowl("closed", "", weigh_bowl())

    # Start main loop
    try:
        main_loop(picam, yolo_model, keras_models, cats_data, bowl, (180, 180))
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
