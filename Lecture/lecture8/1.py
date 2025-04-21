import cv2
import numpy as np

# Load the input image
image_path = 'img/class2.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"The image file '{image_path}' was not found. Please check the file path.")

# Define the homography matrix (example values)
homography_matrix = np.array([
    [1.2, 0.3, 100],
    [0.1, 0.9, 50],
    [0.002, 0.001, 1]
], dtype=np.float32)

# Apply the homography transformation
warped_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))

# Save and display the result
output_path = 'warped_lena.jpg'
cv2.imwrite(output_path, warped_image)
cv2.imshow('Warped Image', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
