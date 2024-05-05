import cv2
import joblib
import numpy as np
from scipy.stats import moment
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from customtkinter import *

# Set the color theme
set_default_color_theme("green")

# Load the pre-trained model
model = joblib.load("steganalysis_model_combined_100.pkl")

# Function to extract YCbCr features from an image
def extract_ycbcr_features(image):
    # Convert image to YCbCr color space
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Extract YCbCr features
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
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply DCT transformation
    dct_image = cv2.dct(np.float32(gray_image))
    # Extract statistical features from DCT coefficients
    mean = np.mean(dct_image)
    std_dev = np.std(dct_image)
    skewness = moment(dct_image.flatten(), moment=3)
    kurtosis = moment(dct_image.flatten(), moment=4)
    # Return extracted features as a numpy array
    return np.array([mean, std_dev, skewness, kurtosis])

# Function to predict if an image is stego or cover given its file path
def predict_image_type(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Extract features
    ycbcr_features = extract_ycbcr_features(image)
    dct_features = extract_dct_features(image)
    features = np.concatenate([ycbcr_features, dct_features]).reshape(1, -1)
    
    # Predict using the pre-trained model
    prediction = model.predict(features)
    
    # Interpret the prediction
    if prediction == 0:
        return "Cover"
    else:
        return "Stego"

# Function to handle image selection
def select_image():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Predict the image type and display the result
        result_label.configure(text="Predicting...")
        image_type = predict_image_type(file_path)
        result_label.configure(text=f"Predicted: {image_type}")

# Function to handle dropping of files
def drop(event):
    # Get the file paths from the dropped files
    file_paths = root.tk.splitlist(event.data)
    # Predict the image types for each dropped file
    results = []
    for file_path in file_paths:
        if os.path.isfile(file_path):
            image_type = predict_image_type(file_path)
            results.append(f"{file_path}: {image_type}")
    # Display the prediction results
    if results:
        result_label.configure(text="\n".join(results))
    else:
        result_label.configure(text="No files dropped")

# Create the main window
root = CTk()
root.title("STEGANALYSIS TOOL")

# Set background color for the root window
root.configure(bg="#606190")

# Create a custom label for the heading
heading_label = CTkLabel(root, text="STEGANALYSIS TOOL", font=("Arial Bold", 20), text_color="#F5F5F5")
heading_label.pack(pady=10)

# Create a custom label for drag and drop
class DragDropLabel(CTkLabel):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.bind("<Button-1>", self.on_click)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.file_paths = []

    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def on_drag(self, event):
        self.place(x=event.x_root - self.start_x, y=event.y_root - self.start_y)

    def on_release(self, event):
        self.place(x=0, y=0)
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.file_paths.append(file_path)
            self.predict_image(file_path)

    def predict_image(self, file_path):
        result_label.configure(text="Predicting...")
        image_type = predict_image_type(file_path)
        result_label.configure(text=f"Predicted: {image_type}")

drag_label = DragDropLabel(root, text="Drag and drop an image here", font=("Arial", 12), text_color="gray", width=40, height=3)
drag_label.pack(pady=10)

# Create a button to select an image
select_button = CTkButton(root, text="Select Image", command=select_image)
select_button.pack(pady=10)

# Create a label to display the result
result_label = CTkLabel(root, text="")
result_label.pack(pady=10)

# Mention that the project is for academic purposes by Harish K
academic_label = CTkLabel(root, text="Project for academic purposes by Harish K", font=("Arial", 10), text_color="gray")
academic_label.pack(pady=10)

# Run the main event loop
root.mainloop()
