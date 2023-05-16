import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from joblib import load

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

import csv
from collections import Counter

from tqdm import tqdm

from visualizer import plot_hero_counts

print("Filtering frames...")
print("Should expect ~50 keyframes to be selected")

video_path = 'overwatch-top-500-05-2023-asia.mp4'
output_file = "stats/" + video_path.replace(".mp4", ".csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_frames_with_abs_diff(video_path, frame_interval=10, threshold=3):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, prev_frame = video.read()
    #prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize a list to store the frames
    selected_frames = []

    # Start processing frames
    frame_count = 1
    while True:
        # Read the next frame
        ret, curr_frame = video.read()
        if not ret:
            break

        # Convert the frame to grayscale
        #curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between frames
        abs_diff = np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32)).mean()

        # Check if the absolute difference is less than the threshold
        if abs_diff < threshold:
            selected_frames.append((frame_count, curr_frame))

        # Update the previous frame
        prev_frame = curr_frame

        # Skip frames according to the specified interval
        for _ in range(frame_interval - 1):
            video.read()

        frame_count += frame_interval

    # Release the video file
    video.release()

    return selected_frames

def plot_frame_differences(selected_frames):
    # Initialize lists to store the time and absolute differences
    times = []
    abs_diffs = []

    # Start processing frames
    frame_count = 1
    for i in range(len(selected_frames) - 1):
        frame_num, curr_frame = selected_frames[i]
        next_frame_num, next_frame = selected_frames[i + 1]

        # Compute the absolute difference between adjacent frames
        abs_diff = np.abs(curr_frame.astype(np.float32) - next_frame.astype(np.float32)).mean()

        # Add the time and absolute difference to the lists
        times.append(frame_count)
        abs_diffs.append(abs_diff)

        frame_count += 1

    # Plot the absolute differences over time
    plt.plot(times, abs_diffs)
    plt.xlabel('Time (frames)')
    plt.ylabel('Absolute Difference')
    plt.title('Absolute Differences between Adjacent Selected Frames')
    plt.show()

def get_frames_with_adjacent_diff(selected_frames, threshold=5):
    # Initialize a list to store the frames
    frames_above_threshold = []

    for i in range(len(selected_frames) - 1):
        frame_num, curr_frame = selected_frames[i]
        next_frame_num, next_frame = selected_frames[i + 1]

        # Compute the absolute difference between adjacent frames
        abs_diff = np.abs(curr_frame.astype(np.float32) - next_frame.astype(np.float32)).mean()

        # Check if the absolute difference is greater than the threshold
        if abs_diff > threshold:
            frames_above_threshold.append((frame_num, curr_frame))
            #frames_above_threshold.append((next_frame_num, next_frame))

    return frames_above_threshold

selected_frames = get_frames_with_abs_diff(video_path, frame_interval=5, threshold=3)
frames_above_threshold = get_frames_with_adjacent_diff(selected_frames, threshold=5)

print(f"{len(frames_above_threshold)} frames selected in initial pass")

def filter_by_frame_num(frames):
    # Remove frames where the frame number has already been seen, or if 
    # the frame number is within 10 frames of a previously selected frame
    seen = set()
    filtered_frames = []
    for frame_num, frame in frames:
        if frame_num not in seen:
            filtered_frames.append((frame_num, frame))
            seen.update(range(frame_num - 10, frame_num + 11))
    return filtered_frames

print("Filtering frames by frame number...")
frames_above_threshold = filter_by_frame_num(frames_above_threshold)
print(f"{len(frames_above_threshold)} frames selected after filtering by frame number")

# Uncomment to plot the absolute differences between adjacent frames
#plot_frame_differences(frames_above_threshold)
print("Final filtering based on cropped differences...")

frames_above_threshold = get_frames_with_adjacent_diff(frames_above_threshold, threshold=4)

print(f"{len(frames_above_threshold)} frames selected in total!")


def show_frames(frames):
    for frame_num, frame in frames:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Frame {}'.format(frame_num))
        plt.axis('off')
        plt.show()

# Uncomment if you want to see what the frames look like
#show_frames(frames_above_threshold)

def crop_frame(frame, x, y, width, height):
    """
    Crop a specified patch from a frame.

    Args:
        frame (numpy.ndarray): The input frame.
        x (int): The x-coordinate of the top-left corner of the patch.
        y (int): The y-coordinate of the top-left corner of the patch.
        width (int): The width of the patch.
        height (int): The height of the patch.

    Returns:
        numpy.ndarray: The cropped patch.
    """
    return frame[y:y+height, x:x+width]

def visualize_cropping(frame, x, y, width, height):
    """
    Visualize the specified cropping area on a frame.

    Args:
        frame (numpy.ndarray): The input frame.
        x (int): The x-coordinate of the top-left corner of the patch.
        y (int): The y-coordinate of the top-left corner of the patch.
        width (int): The width of the patch.
        height (int): The height of the patch.
    """
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


# Uncomment to visualize cropping
#frame = frames_above_threshold[0][1]
#visualize_cropping(frame, 1303, 301, 159, 548)

cropped_frames = [(frame_num, crop_frame(frame, 1305, 299, 159, 548)) for frame_num, frame in frames_above_threshold]

patch_size = (55, 54)  # (height, width)

def split_image_into_patches(image, patch_size):
    """
    Split an image into patches.

    Args:
        image (numpy.ndarray): The input image.
        patch_size (tuple): The size of the patch (height, width).

    Returns:
        list: List of patches.
    """
    height, width = image.shape[:2]
    patch_height, patch_width = patch_size

    patches = []
    for y in range(0, height, patch_height):
        for x in range(0, width, patch_width):
            patch = image[y:y+patch_height, x:x+patch_width]
            patches.append(patch)

    return patches

# Uncomment to show cropped frames
# show_frames(cropped_frames)

print("Splitting frames into patches...")
patch_list = [(frame_num, patch) for frame_num, frame in cropped_frames for patch in split_image_into_patches(frame, patch_size)]

# Uncomment to show patches
#show_frames(patch_list)


# DONE: In hand-labeler.py 
# Uncomment the following code if you need to label new patches/heros
#patch_num = 0
#for frame_num, patch in patch_list:
#    cv2.imwrite(f"patch_dataset/unlabeled/{patch_num}.png", patch)
#    patch_num += 1

# DONE: Train a model to classify patches (kNN with CLIP features, k=1)

def get_clip_features(patch):
    with torch.no_grad():
        inputs = processor(images=patch, return_tensors="pt").to(device)
        if device == "cpu":
            return model.get_image_features(**inputs).detach().numpy().flatten()
        else:
            return model.get_image_features(**inputs).cpu().detach().numpy().flatten()

def classify_patch(clf, patch):
    """
    Classify a patch.

    Args:
        patch (numpy.ndarray): The input patch.

    Returns:
        str: The predicted class.
    """
    # Convert patch to RGB
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    # Convert patch to CLIP features
    features = get_clip_features(patch_rgb)

    # Classify the patch using the model
    return clf.predict([features])[0]

clf = load("models/dps_classifier.joblib")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

preds = []
print("Classifying each hero patch...")
for frame_num, patch in tqdm(patch_list):
    # Convert patch to CLIP features
    pred = classify_patch(clf, patch)
    preds.append(pred)

# TODO: Accumulate results to csv file
def create_csv_with_counts(strings, output_file):
    # Count the occurrences of each string
    string_counts = Counter(strings)

    # Open the CSV file in write mode
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['Hero', 'Count'])

        # Write each string and its count as a row in the CSV file
        for string, count in string_counts.items():
            if string != "nothing":
                writer.writerow([string, count])


create_csv_with_counts(preds, output_file)
plot_hero_counts(output_file)