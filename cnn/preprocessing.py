import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define the custom image processing function
minValue = 70

def func(path):
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res

# Main code for processing images and splitting into train/test sets
if __name__ == "__main__":
    # Specify the input directory containing the images
    input_dir = r"data\train"
    
    # Specify the output directory for processed images
    output_dir = "processed_images"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize lists to store file paths and labels
    file_paths = []
    labels = []

    # Process each image in the input directory
    for label, label_name in enumerate(os.listdir(input_dir)):
        label_dir = os.path.join(input_dir, label_name)
        if os.path.isdir(label_dir):
            for root, _, files in os.walk(label_dir):
                for filename in files:
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        input_path = os.path.join(root, filename)
                        output_path = os.path.join(output_dir, os.path.relpath(input_path, input_dir))
                        output_dirname = os.path.dirname(output_path)
                        
                        # Create output directory if it doesn't exist
                        if not os.path.exists(output_dirname):
                            os.makedirs(output_dirname)
                        
                        # Apply custom image processing function
                        processed_image = func(input_path)
                        
                        # Write processed image to output directory
                        cv2.imwrite(output_path, processed_image)
                        
                        # Append file path and label to lists
                        file_paths.append(output_path)
                        labels.append(label)

                        print(f"Processed: {input_path} -> {output_path}")
