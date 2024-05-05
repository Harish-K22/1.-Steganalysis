import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import moment
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Function to extract YCbCr features from an image
def extract_ycbcr_features(image):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycbcr_features = [
        np.mean(ycbcr_image[:, :, 0]),  # Mean of Y channel
        np.mean(ycbcr_image[:, :, 1]),  # Mean of Cb channel
        np.mean(ycbcr_image[:, :, 2]),  # Mean of Cr channel
        np.std(ycbcr_image[:, :, 0]),   # Standard deviation of Y channel
        np.std(ycbcr_image[:, :, 1]),   # Standard deviation of Cb channel
        np.std(ycbcr_image[:, :, 2])    # Standard deviation of Cr channel
    ]
    return ycbcr_features

# Function to extract DCT features from an image
def extract_dct_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct_image = cv2.dct(np.float32(gray_image))
    mean = np.mean(dct_image)
    std_dev = np.std(dct_image)
    skewness = moment(dct_image.flatten(), moment=3)
    kurtosis = moment(dct_image.flatten(), moment=4)
    return np.array([mean, std_dev, skewness, kurtosis])

# Directory paths
cover_dir = "Cover"
stego_dir = "JMiPOD"

# Initialize lists to store features and labels
ycbcr_features = []
dct_features = []
labels = []

# Define image augmentation function using OpenCV
def augment_image(image):
    augmented_images = []
    for angle in [-20, 20]:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        augmented_images.append(rotated_image)
    return augmented_images

# Process cover images
for file in tqdm(os.listdir(cover_dir), desc="Processing Cover Images"):
    image_path = os.path.join(cover_dir, file)
    image = cv2.imread(image_path)
    ycbcr_features.append(extract_ycbcr_features(image))
    dct_features.append(extract_dct_features(image))
    labels.append(0)  # Label 0 for cover images
    # Apply data augmentation
    augmented_images = augment_image(image)
    for augmented_image in augmented_images:
        ycbcr_features.append(extract_ycbcr_features(augmented_image))
        dct_features.append(extract_dct_features(augmented_image))
        labels.append(0)

# Process stego images
for file in tqdm(os.listdir(stego_dir), desc="Processing Stego Images"):
    image_path = os.path.join(stego_dir, file)
    image = cv2.imread(image_path)
    ycbcr_features.append(extract_ycbcr_features(image))
    dct_features.append(extract_dct_features(image))
    labels.append(1)  # Label 1 for stego images
    # Apply data augmentation
    augmented_images = augment_image(image)
    for augmented_image in augmented_images:
        ycbcr_features.append(extract_ycbcr_features(augmented_image))
        dct_features.append(extract_dct_features(augmented_image))
        labels.append(1)

# Convert lists to arrays
ycbcr_features = np.array(ycbcr_features)
dct_features = np.array(dct_features)
labels = np.array(labels)

# Save features and labels to CSV file
ycbcr_df = pd.DataFrame(ycbcr_features, columns=["Y_Mean", "Cb_Mean", "Cr_Mean", "Y_Std_Dev", "Cb_Std_Dev", "Cr_Std_Dev"])
dct_df = pd.DataFrame(dct_features, columns=["DCT_Mean", "DCT_Std_Dev", "DCT_Skewness", "DCT_Kurtosis"])
df = pd.concat([ycbcr_df, dct_df], axis=1)
df["Label"] = labels
csv_file = "image_features_increase_accuracy_72_fullimgs.csv"
df.to_csv(csv_file, index=False)
print(f"Features saved to {csv_file}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["Label"]), df["Label"], test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the trained model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save the trained model
model_file = "steganalysis_model_increase_accuracy_72_fullimgs.pkl"
joblib.dump(model, model_file)
print(f"Model saved to {model_file}")
