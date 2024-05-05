Methodology
•	Data Collection and Preprocessing:
o	Data Collection: I acquired a broad collection of cover and stego photos from numerous sources to assure representation at all resolutions, formats, and compression levels. The dataset was taken via the Kaggle Alaska2 competition, which supplied photos processed using three different steganographic methods: JMiPOD, UERD, and JUNIWARD, each with 75,000 images.
For further exploration or replication, the dataset can be accessed here.
o	Preprocessing: Uniformly preprocess the images by applying techniques such as resizing, normalization, and data augmentation to enhance the diversity and robustness of the dataset.
•	Feature Extraction:
At the start of the project, getting the features from the images was tough. It took a lot of time and the computer's memory struggled with it. So, I tried different ways to get these features faster and with less memory. Eventually, I found that using DCT and YCbCr methods worked best for me.
o	YCbCr Features: Extracted YCbCr features from the images, including mean and standard deviation of each channel, to capture color space characteristics.
o	DCT Features: Calculated statistical features from the Discrete Cosine Transform (DCT) coefficients of grayscale images, including mean, standard deviation, skewness, and kurtosis.
•	Model Training:
o	Random Forest Classifier: Implemented a Random Forest classifier for steganalysis due to its robustness and suitability for classification tasks.
o	Training Process: Split the dataset into training and testing sets (80/20 ratio) and trained the Random Forest classifier using the extracted features.
•	Model Evaluation:
o	Accuracy Calculation: Evaluated the trained model's performance on the testing set, calculating accuracy to measure its effectiveness in distinguishing between cover and stego images.
o	Achieved Accuracy: The trained Random Forest classifier achieved an accuracy of 72% on the testing dataset, indicating its capability to differentiate between cover and stego images.
•	Model Deployment and Integration:
o	Deployment: Deployed the trained Random Forest classifier into a user-friendly software application using tkinter, enabling users to predict the image type interactively.
o	Integration: Ensured compatibility across different operating systems and environments for seamless integration into existing workflows.
