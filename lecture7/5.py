import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = 'img/dog.jpg'  # Change path to your image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not read the image at {image_path}. Check the path!")

# Define points in the original and transformed images
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

# Compute affine transformation matrix and apply it
M = cv2.getAffineTransform(pts1, pts2)
affine_transformed = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# Display the original and transformed images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(affine_transformed, cv2.COLOR_BGR2RGB))
axes[1].set_title('Affine Transformed Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
