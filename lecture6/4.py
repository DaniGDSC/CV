import cv2
import numpy as np
import matplotlib.pyplot as plt

# Compute gradients and angles
def compute_gradients(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.mod(np.arctan2(dy, dx) * (180 / np.pi), 360)  # Ensure angles are in [0, 360]
    return magnitude, angle

# Determine the dominant orientation of a keypoint
def compute_keypoint_orientation(magnitude, angle, x, y, patch_size=16):
    half_patch = patch_size // 2
    region_mag = magnitude[max(0, y-half_patch):y+half_patch, max(0, x-half_patch):x+half_patch]
    region_angle = angle[max(0, y-half_patch):y+half_patch, max(0, x-half_patch):x+half_patch]
    hist, _ = np.histogram(region_angle, bins=36, range=(0, 360), weights=region_mag)
    return np.argmax(hist) * (360 / 36)

# Extract and rotate patch
def extract_patch(image, x, y, orientation, patch_size=16):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    half_patch = patch_size // 2
    patch = gray[max(0, y-half_patch):min(gray.shape[0], y+half_patch),
                 max(0, x-half_patch):min(gray.shape[1], x+half_patch)]
    patch = cv2.resize(patch, (patch_size, patch_size)) if patch.shape[:2] != (patch_size, patch_size) else patch
    center = (patch_size // 2, patch_size // 2)
    M = cv2.getRotationMatrix2D(center, orientation, 1.0)
    return cv2.warpAffine(patch, M, (patch_size, patch_size))

# Compute histogram for a cell
def compute_cell_histogram(magnitude, angle, cell_size=4, n_bins=8):
    bin_width = 360 / n_bins
    hist = np.zeros(n_bins)
    for i in range(cell_size):
        for j in range(cell_size):
            mag, ang = magnitude[i, j], angle[i, j]
            bin_idx = int(ang // bin_width)
            fraction = (ang % bin_width) / bin_width
            hist[bin_idx] += mag * (1 - fraction)
            hist[(bin_idx + 1) % n_bins] += mag * fraction
    return hist

# Compute SIFT descriptor
def compute_sift_descriptor(image, x, y):
    magnitude, angle = compute_gradients(image)
    orientation = compute_keypoint_orientation(magnitude, angle, x, y)
    patch = extract_patch(image, x, y, orientation)
    dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    patch_mag = np.sqrt(dx**2 + dy**2)
    patch_angle = np.mod(np.arctan2(dy, dx) * (180 / np.pi), 360)
    descriptor = []
    cell_size, n_bins = 4, 8
    for cy in range(4):
        for cx in range(4):
            y_start, x_start = cy * cell_size, cx * cell_size
            cell_mag = patch_mag[y_start:y_start+cell_size, x_start:x_start+cell_size]
            cell_angle = patch_angle[y_start:y_start+cell_size, x_start:x_start+cell_size]
            descriptor.extend(compute_cell_histogram(cell_mag, cell_angle, cell_size, n_bins))
    descriptor = np.array(descriptor)
    descriptor /= (np.linalg.norm(descriptor) + 1e-6)
    descriptor = np.clip(descriptor, 0, 0.2)
    descriptor /= (np.linalg.norm(descriptor) + 1e-6)
    return descriptor

# Read image
image = cv2.imread('img/dog.jpg')
if image is None:
    print("Cannot read image. Using a placeholder.")
    image = np.zeros((256, 256, 3), dtype=np.uint8)

# Select keypoint
h, w = image.shape[:2]
x, y = w // 2, h // 2

# Compute SIFT descriptor
sift_descriptor = compute_sift_descriptor(image, x, y)

# Display image with keypoint
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter(x, y, c='red', s=100, marker='x')
plt.title('Original Image with Keypoint')
plt.axis('off')
plt.show()

# Print descriptor
print("SIFT Descriptor:")
print(sift_descriptor)
print("Descriptor size:", sift_descriptor.shape)

# Plot descriptor
plt.figure(figsize=(10, 4))
plt.bar(range(len(sift_descriptor)), sift_descriptor)
plt.title('SIFT Descriptor (128 dimensions)')
plt.xlabel('Feature Index')
plt.ylabel('Normalized Magnitude')
plt.show()
