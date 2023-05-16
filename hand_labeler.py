import os
import shutil
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk


# Define the category names
category_names = [
    "Ashe", "Tracer", "Soldier76", "Widowmaker", "Cassidy", "Junkrat", "Mei",
    "Reaper", "Echo", "Hanzo", "Sojourn", "Pharah", "Sombra", "Symmetra",
    "Bastion", "Torbjorn", "Genji", "Nothing"
]

category_names = [name.lower() for name in category_names]

# Path to the directory containing the images
images_directory = "patch_dataset/unlabeled"

# Path to the directory where categorized images will be moved
output_directory = "patch_dataset/labeled"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Initialize variables
current_image_index = 0

# Function to handle the image labeling process
def label_image(category_index):
    global current_image_index
    
    # Get the category name based on the index
    category = category_names[category_index]
    
    # Move the image file to the category directory
    image_file = image_files[current_image_index]
    image_path = os.path.join(images_directory, image_file)
    category_directory = os.path.join(output_directory, category)
    os.makedirs(category_directory, exist_ok=True)
    new_image_path = os.path.join(category_directory, image_file)
    shutil.move(image_path, new_image_path)
    
    # Update the current image index
    current_image_index += 1
    
    # Check if all images have been labeled
    if current_image_index >= len(image_files):
        label.config(text="Labeling complete.")
        button.config(state="disabled")
        image_label.config(image="")
    else:
        # Display the next image
        next_image_path = os.path.join(images_directory, image_files[current_image_index])
        display_image(next_image_path)
        label.config(text=f"Labeling image: {image_files[current_image_index]}")

# Function to display the image
def display_image(image_path):
    image = Image.open(image_path)
    image = image.resize((300, 300))  # Resize the image as needed
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

# Create the main window
root = Tk()
root.title("Image Labeling Tool")

# Get a list of image files in the images directory
image_files = [file for file in os.listdir(images_directory) if file.endswith((".jpg", ".png"))]

# Create the label widget to display the image
image_label = Label(root)
image_label.pack()

# Create the label widget to display the image filename
label = Label(root, text=f"Labeling image: {image_files[current_image_index]}")
label.pack()

# Create a button for each category
for index, category in enumerate(category_names):
    button = Button(root, text=category, command=lambda idx=index: label_image(idx))
    button.pack()

# Display the first image
first_image_path = os.path.join(images_directory, image_files[current_image_index])
display_image(first_image_path)

# Start the GUI event loop
root.mainloop()

import os
import matplotlib.pyplot as plt

def plot_histogram(categorized_images_directory):
    category_counts = {}
    
    # Get a list of subdirectories (categories)
    categories = [category for category in os.listdir(categorized_images_directory) if os.path.isdir(os.path.join(categorized_images_directory, category))]
    
    # Count the number of images in each category
    for category in categories:
        category_directory = os.path.join(categorized_images_directory, category)
        image_files = [file for file in os.listdir(category_directory) if file.endswith((".jpg", ".png"))]
        category_counts[category] = len(image_files)
    
    # Sort the category counts dictionary by category name
    category_counts_sorted = sorted(category_counts.items(), key=lambda x: x[0])
    categories, counts = zip(*category_counts_sorted)
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts)
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.title("Hero Distribution")
    plt.xticks(rotation=45)

    # Add labels with counts above each bar
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center')

    plt.tight_layout()
    plt.show()

plot_histogram(output_directory)