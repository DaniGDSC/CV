import cv2
import numpy as np

# Load the image in grayscale
image_path = 'img/dog.jpg'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load the image. Please check the file path.")
    exit()

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(image, None)

# Print the number of keypoints detected
print(f"Number of keypoints detected: {len(keypoints)}")

# Draw keypoints on the image with rich keypoint flags (includes orientation and scale)
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures circles with orientation are drawn
image_with_keypoints = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Display the image with keypoints
cv2.imshow("SIFT Keypoints", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
