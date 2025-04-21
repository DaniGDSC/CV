import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin


def load_visual_vocabulary(vocabulary_path):
    """Load the precomputed visual vocabulary."""
    return np.load(vocabulary_path)


def load_test_image(image_path):
    """Load the test image in grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Error: Could not load the test image at {image_path}.")
    return image


def extract_sift_descriptors(image):
    """Extract SIFT descriptors from the given image."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None or len(descriptors) == 0:
        raise ValueError("Error: No descriptors found in the test image.")
    return descriptors


def compute_histogram(descriptors, visual_vocabulary):
    """Compute the histogram of visual words."""
    k = visual_vocabulary.shape[0]
    labels = pairwise_distances_argmin(descriptors, visual_vocabulary)
    histogram = np.zeros(k, dtype=np.float32)
    for label in labels:
        histogram[label] += 1
    return histogram / np.sum(histogram)  # Normalize the histogram


def visualize_histogram(histogram):
    """Visualize the histogram as a bar plot."""
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(histogram)), histogram, width=1.0, align='center')
    plt.title("Bag-of-Words Histogram")
    plt.xlabel("Visual Word Index")
    plt.ylabel("Normalized Frequency")
    plt.show()


def main():
    visual_vocabulary_path = "Database/visual_vocabulary.npy"
    test_image_path = "img/dog3.jpg"

    try:
        visual_vocabulary = load_visual_vocabulary(visual_vocabulary_path)
        test_image = load_test_image(test_image_path)
        descriptors = extract_sift_descriptors(test_image)
        histogram = compute_histogram(descriptors, visual_vocabulary)
        visualize_histogram(histogram)

        # Optionally, print the histogram
        print("Normalized Histogram:", histogram)
        print("Sum of histogram (should be 1):", np.sum(histogram))
    except (FileNotFoundError, ValueError) as e:
        print(e)


if __name__ == "__main__":
    main()