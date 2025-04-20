import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def load_images(image_paths):
    """Load images in grayscale and return a list of descriptors."""
    sift = cv2.SIFT_create()
    all_descriptors = []

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Could not load image {image_path}. Skipping...")
            continue

        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)

    return all_descriptors

def stack_descriptors(descriptors_list):
    """Stack all descriptors into a single array."""
    if not descriptors_list:
        raise ValueError("No descriptors found in any of the images.")
    return np.vstack(descriptors_list)

def create_visual_vocabulary(descriptors, k, random_state=42):
    """Apply K-means clustering to create the visual vocabulary."""
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_

def save_visual_vocabulary(vocabulary, output_path):
    """Save the visual vocabulary to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, vocabulary)
    print(f"Visual vocabulary saved as '{output_path}'")

def main():
    # List of training image paths
    image_paths = [
        'img/dog.jpg',
        'img/dog1.jpg',
        'img/dog2.jpg',
        'img/dog3.jpg',
        'img/dog4.jpg',
        'img/dog5.jpg',
        'img/dog6.jpg',
        'img/dog7.jpg'
    ]

    # Parameters
    k = 100  # Number of clusters (visual words)
    output_path = "Database/visual_vocabulary.npy"

    # Step 1: Load images and extract SIFT descriptors
    all_descriptors = load_images(image_paths)

    # Step 2: Stack all descriptors into a single array
    try:
        all_descriptors = stack_descriptors(all_descriptors)
        print(f"Total number of descriptors: {all_descriptors.shape[0]}")
    except ValueError as e:
        print(e)
        return

    # Step 3: Create the visual vocabulary
    visual_vocabulary = create_visual_vocabulary(all_descriptors, k)
    print(f"Shape of visual vocabulary: {visual_vocabulary.shape}")

    # Step 4: Save the visual vocabulary
    save_visual_vocabulary(visual_vocabulary, output_path)

if __name__ == "__main__":
    main()