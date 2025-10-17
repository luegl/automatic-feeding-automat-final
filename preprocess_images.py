"""
Cat Image Preprocessing Pipeline
--------------------------------
This program detects cats in images using YOLO, crops the detected
cats, saves them in a preprocessed folder, and balances the number
of images between the target cat and other cats.
"""

# Install YOLO 
!pip install ultralytics

# Standard library
import os
import random
import shutil

# Third-party libraries
import cv2
from google.colab import drive
from ultralytics import YOLO



# -------------------------------------------------------------------
#   Constants
# -------------------------------------------------------------------

CAT = "bruno"
MODEL = YOLO("yolov8n.pt")
DATASET_NAME = f"dataset_{CAT[:3]}_other"
INPUT_FOLDER = f"/content/drive/MyDrive/datasets/{DATASET_NAME}/images"
OUTPUT_FOLDER = f"/content/drive/MyDrive/datasets/{DATASET_NAME}/preprocessed_images"
CLASS_FOLDERS = [CAT, "other_cats"]

# -------------------------------------------------------------------
#   Functions
# -------------------------------------------------------------------

def mount_drive():
    """Mount Google Drive in Colab to access the image dataset."""
    drive.mount('/content/drive')


def create_output_dirs():
    """Create output directories for each class (cat and other cats)."""
    for category in CLASS_FOLDERS:
        os.makedirs(os.path.join(OUTPUT_FOLDER, category), exist_ok=True)


def process_images():
    """
    Detect cats in images using YOLO, crop detected cats, and save
    them in the preprocessed folder. Remove images without detected cats.
    """
    for class_name in CLASS_FOLDERS:
        class_path = os.path.join(INPUT_FOLDER, class_name)
        start_index = 0

        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)

                # Detect cats using YOLO
                results = MODEL.predict(source=img_path, conf=0.47, classes=[15])

                if results[0].boxes:
                    boxes = results[0].boxes.xyxy.tolist()
                    for box in boxes:
                        x_min, y_min, x_max, y_max = map(int, box)
                        cropped_img = img[y_min:y_max, x_min:x_max]

                        # Save cropped image
                        cropped_img_path = os.path.join(
                            OUTPUT_FOLDER, class_name, f"{class_name}_{start_index}.jpg"
                        )
                        cv2.imwrite(cropped_img_path, cropped_img)
                        start_index += 1
                else:
                    # Remove images without detected cats
                    os.remove(img_path)


def balance_dataset():
    """
    Balance the number of images between CAT and other_cats by removing
    excess files or duplicating images to match the smaller class.
    """
    cat_files = [
        f for f in os.listdir(os.path.join(OUTPUT_FOLDER, CAT))
        if f.endswith((".jpg", ".png"))
    ]
    other_cats_files = [
        f for f in os.listdir(os.path.join(OUTPUT_FOLDER, "other_cats"))
        if f.endswith((".jpg", ".png"))
    ]

    cat_count = len(cat_files)
    other_cats_count = len(other_cats_files)

    if cat_count != other_cats_count:
        target_count = min(cat_count, other_cats_count)

        # Remove excess files from the larger class
        if cat_count > target_count:
            for file in random.sample(cat_files, cat_count - target_count):
                os.remove(os.path.join(OUTPUT_FOLDER, CAT, file))
        elif other_cats_count > target_count:
            for file in random.sample(other_cats_files, other_cats_count - target_count):
                os.remove(os.path.join(OUTPUT_FOLDER, "other_cats", file))

        # Refresh lists after removal
        cat_files = [f for f in os.listdir(os.path.join(OUTPUT_FOLDER, CAT)) if f.endswith((".jpg", ".png"))]
        other_cats_files = [f for f in os.listdir(os.path.join(OUTPUT_FOLDER, "other_cats")) if f.endswith((".jpg", ".png"))]

        # Duplicate files to balance smaller class
        if cat_count < other_cats_count:
            for _ in range(target_count - cat_count):
                file_to_copy = random.choice(other_cats_files)
                shutil.copy(
                    os.path.join(OUTPUT_FOLDER, "other_cats", file_to_copy),
                    os.path.join(OUTPUT_FOLDER, CAT, file_to_copy)
                )
        elif other_cats_count < cat_count:
            for _ in range(target_count - other_cats_count):
                file_to_copy = random.choice(cat_files)
                shutil.copy(
                    os.path.join(OUTPUT_FOLDER, CAT, file_to_copy),
                    os.path.join(OUTPUT_FOLDER, "other_cats", file_to_copy)
                )


# -------------------------------------------------------------------
#   Main
# -------------------------------------------------------------------

def main():
    """Run the full image preprocessing pipeline."""
    mount_drive()
    create_output_dirs()
    process_images()
    balance_dataset()
    print("Images have been processed, balanced, and saved.")


if __name__ == "__main__":
    main()
