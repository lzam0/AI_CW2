import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Path to ASL images
IMAGE_DIR = "data/CW2_dataset_final"
MODEL_PATH = "models/hand_landmarker.task"
print(MODEL_PATH)
OUTPUT_CSV = "data/extracted_features/hand_landmarks.csv"

# Create HandLandmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# CSV Headers

header = ["instance_id"]

for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])

header.append("label")

# Ensure that the output CSV dir is valid
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Write CSV
with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    instance_id = 0

    # Loop through ASL labels (A–J)
    for label in sorted(os.listdir(IMAGE_DIR)):
        label_path = os.path.join(IMAGE_DIR, label)

        # Skip anything that isn't a folder
        if not os.path.isdir(label_path):
            continue

        # Loop through images in each label folder
        for filename in os.listdir(label_path):

            # Process only .jpg images
            if not filename.lower().endswith(".jpg"):
                continue

            image_path = os.path.join(label_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                continue

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )

            result = detector.detect(mp_image)

            # Treat failed detections as noise
            if not result.hand_landmarks:
                continue

            landmarks = result.hand_landmarks[0]

            row = [instance_id]

            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z])

            row.append(label)
            writer.writerow(row)

            instance_id += 1


print("Feature extraction complete.")
print(f"Saved to: {OUTPUT_CSV}")

# --------------------------------------------------------------------------------------------
# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Load the extracted features dataset
data = pd.read_csv('data/extracted_features/hand_landmarks.csv')

# Separate features and labels
X = data.drop(columns=['instance_id', 'label']).values
y = data['label'].values

# Create training testing set - random state utilised so that the split is reproducible (seed)
# MUST SPLIT INTO 60-20-20 LATER FOR TRAIN-VAL-TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Create visualisations of the training data - scatter plot of first two features (x0, y0)
plt.figure(figsize=(10, 6))

unique_labels = np.unique(y_train)
for label in unique_labels:
    mask = y_train == label
    plt.scatter(X_train[mask, 0], X_train[mask, 1], label=f'Sign {label}', alpha=0.6)

plt.title('Scatter Plot of ASL Hand Landmarks (Feature x0 vs y0)')
plt.xlabel('Wrist X-coordinate (Feature 1)')
plt.ylabel('Wrist Y-coordinate (Feature 2)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside the plot
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()
