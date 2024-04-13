import os
import shutil
from sklearn.model_selection import train_test_split

# Specify the path to your dataset directory
dataset_dir = 'processed_images'

# List all subdirectories (labels)
labels = [label for label in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, label))]

# Create directories for the training and testing sets
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split the dataset into training and testing sets while preserving the folder structure
for label in labels:
    label_dir = os.path.join(dataset_dir, label)
    train_label_dir = os.path.join(train_dir, label)
    test_label_dir = os.path.join(test_dir, label)
    
    # Create subdirectories for the label in the training and testing sets
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    
    # List all image files for the label
    image_files = [file for file in os.listdir(label_dir) if file.endswith('.jpg') or file.endswith('.png')]
    
    # Split the image files into training and testing sets
    train_files, test_files = train_test_split(image_files, test_size=0.25, random_state=42)
    
    # Move training set images to the train directory
    for file in train_files:
        src = os.path.join(label_dir, file)
        dst = os.path.join(train_label_dir, file)
        shutil.copy(src, dst)
    
    # Move testing set images to the test directory
    for file in test_files:
        src = os.path.join(label_dir, file)
        dst = os.path.join(test_label_dir, file)
        shutil.copy(src, dst)
