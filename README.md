
Steganalysis Project
Overview
This repository contains code and documentation for a steganalysis project aimed at distinguishing between cover and stego images using machine learning techniques.

Methodology
Data Collection and Preprocessing
Data Collection: Acquired a diverse collection of cover and stego photos from various sources, including the Kaggle Alaska2 competition. The dataset consists of images processed using three different steganographic methods: JMiPOD, UERD, and JUNIWARD, each with 75,000 images.
Preprocessing: Uniformly preprocessed the images by applying techniques such as resizing, normalization, and data augmentation to enhance dataset diversity and robustness.
Feature Extraction
YCbCr Features: Extracted YCbCr features from the images, including mean and standard deviation of each channel, to capture color space characteristics.
DCT Features: Calculated statistical features from the Discrete Cosine Transform (DCT) coefficients of grayscale images, including mean, standard deviation, skewness, and kurtosis.
Model Training
Random Forest Classifier: Implemented a Random Forest classifier due to its robustness and suitability for classification tasks.
Training Process: Split the dataset into training and testing sets (80/20 ratio) and trained the Random Forest classifier using the extracted features.
Model Evaluation
Accuracy Calculation: Evaluated the trained model's performance on the testing set, calculating accuracy to measure its effectiveness in distinguishing between cover and stego images.
Achieved Accuracy: The trained Random Forest classifier achieved an accuracy of 72% on the testing dataset, indicating its capability to differentiate between cover and stego images.
Model Deployment and Integration
Deployment: Deployed the trained Random Forest classifier into a user-friendly software application using tkinter, enabling users to predict the image type interactively.
Integration: Ensured compatibility across different operating systems and environments for seamless integration into existing workflows.
Usage
Dependencies: Ensure all necessary dependencies are installed by running pip install -r requirements.txt.
Data Preparation: Prepare your dataset or use the provided Kaggle Alaska2 dataset.
Training: Run train_model.py to train the Random Forest classifier.
Testing: Evaluate the model's performance by running test_model.py.
Deployment: Deploy the model using deploy_model.py for interactive prediction.
References
Kaggle Alaska2 competition dataset: link
For further details and access to the dataset, refer to the project documentation and repository files.
