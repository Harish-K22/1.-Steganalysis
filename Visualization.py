import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    return np.array(Image.open(image_path))

def calculate_differences(image1, image2):
    return np.abs(image1 - image2)

def visualize_differences(differences):
    plt.figure(figsize=(8, 6))
    plt.imshow(differences, cmap='gray')
    plt.title('Absolute Differences')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def detect_stego(image_path_cover, image_path_stego):
    # Load cover and stego images
    cover_image = load_image(image_path_cover)
    stego_image = load_image(image_path_stego)
    
    # Calculate absolute differences
    differences = calculate_differences(cover_image, stego_image)
    
    # Visualize differences
    visualize_differences(differences)
    
    # Detection mechanism based on the ratio of average difference to maximum possible difference
    avg_difference = np.mean(differences)
    max_possible_difference = np.max(stego_image) - np.min(stego_image)
    difference_ratio = avg_difference / max_possible_difference
    
    if difference_ratio > 0:  # Adjust this threshold as needed
        print("The image is likely to be stego.")
    else:
        print("The image is likely to be cover.")

if __name__ == "__main__":
    cover_image_path = r"C:\Users\mshar\OneDrive\Documents\VIT\Data Privacy\Cover\00002.jpg"
    #stego_image_path = r"C:\Users\mshar\OneDrive\Documents\VIT\Data Privacy\Cover\00002.jpg"
    #stego_image_path = r"C:\Users\mshar\OneDrive\Documents\VIT\Data Privacy\UERD\00002.jpg"
    #stego_image_path = r"C:\Users\mshar\OneDrive\Documents\VIT\Data Privacy\JMiPOD\00002.jpg"
    stego_image_path = r"C:\Users\mshar\OneDrive\Documents\VIT\Data Privacy\JUNIWARD\00002.jpg"
    
    detect_stego(cover_image_path, stego_image_path)
