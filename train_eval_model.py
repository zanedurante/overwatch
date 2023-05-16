import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL import Image
import torch

from tqdm import tqdm
import pickle

from joblib import dump, load

# Path to the labeled image dataset
dataset_path = "patch_dataset/labeled"

FEATURE_TYPE = "CLIP"

# Pre-load model
if FEATURE_TYPE == "CLIP":
    # Load CLIP model
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    def get_clip_features(path):
        image = Image.open(path)
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            return model.get_image_features(**inputs).detach().numpy().flatten()


# Create empty lists to store the training and test data
train_data = []
train_labels = []
test_data = []
test_labels = []

def get_features(path, feature_type="CLIP"):
    if feature_type == "histogram":
        image = cv2.imread(path)
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        print(hist.shape)
        return hist.flatten()
    elif feature_type == "CLIP":
        return get_clip_features(path)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

# Iterate through each class in the dataset
print("Constructing dataset...")
for class_name in tqdm(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        image_files = os.listdir(class_path)
        
        # Take the first image from each class as training data
        train_hist = get_features(os.path.join(class_path, image_files[0]))
        train_data.append(train_hist)
        train_labels.append(class_name)
        
        # Use the rest of the images as test data
        for i in range(1, len(image_files)):
            test_hist = get_features(os.path.join(class_path, image_files[i]))
            test_data.append(test_hist)
            test_labels.append(class_name)

# Convert the lists to NumPy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Create and train the kNN classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)
print(train_labels)
knn.fit(train_data, train_labels)

dump(knn, "models/dps_classifier.joblib")

# Load model to test that it is working
knn = load("models/dps_classifier.joblib")


# Perform classification on the test data
predicted_labels = knn.predict(test_data)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy (%):", accuracy * 100)

def get_incorrect_predictions(predicted_labels, test_labels):
    incorrect_predictions = []
    for i in range(len(predicted_labels)):
        if predicted_labels[i] != test_labels[i]:
            incorrect_predictions.append((predicted_labels[i], test_labels[i]))
    return incorrect_predictions

incorrect_predictions = get_incorrect_predictions(predicted_labels, test_labels)

for prediction in incorrect_predictions:
    predicted, actual = prediction
    print("Predicted:", predicted, "  Actual:", actual)