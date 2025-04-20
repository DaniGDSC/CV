import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Exercise 6: Real-world Two-view Geometry ---
print("=== Real-world Two-view Geometry ===")

# Step 1: Load the two overlapping images
img1 = cv2.imread('img/class1.jpg') 
img2 = cv2.imread('img/class2.jpg')

if img1 is None or img2 is None:
    print("Error: Could not load one or both images.")
    exit()

# Convert images to grayscale for feature detection
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Step 2: Detect and match features using SIFT
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Match features using FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = []
pts1 = []
pts2 = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

# Draw matches (for visualization)
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title('SIFT Feature Matches')
plt.axis('off')
plt.show()

# Step 3: Compute the fundamental matrix F using RANSAC
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)

# Filter matches using the mask from RANSAC
inliers1 = []
inliers2 = []
for i in range(len(mask)):
    if mask[i]:
        inliers1.append(pts1[i])
        inliers2.append(pts2[i])

inliers1 = np.array(inliers1)
inliers2 = np.array(inliers2)

print("Fundamental Matrix F:")
print(F)

# Step 4: Draw epipolar lines
# Compute epipolar lines in the second image for points in the first image
lines2 = cv2.computeCorrespondEpilines(inliers1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)

# Compute epipolar lines in the first image for points in the second image
lines1 = cv2.computeCorrespondEpilines(inliers2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)

# Function to draw epipolar lines
def draw_epipolar_lines(img, lines, points, color):
    img_copy = img.copy()
    h, w = img.shape[:2]
    for line, pt in zip(lines, points):
        a, b, c = line
        # Line equation: ax + by + c = 0
        # At x = 0: y = -c/b
        # At x = w-1: y = -(a*(w-1) + c)/b
        if abs(b) > 1e-10:
            y0 = -c / b
            y1 = -(a * (w - 1) + c) / b
            pt1 = (0, int(y0))
            pt2 = (w - 1, int(y1))
            cv2.line(img_copy, pt1, pt2, color, 1)
            cv2.circle(img_copy, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    return img_copy

# Draw epipolar lines on both images
img1_lines = draw_epipolar_lines(img1, lines1, inliers1, (255, 0, 0))  # Blue lines
img2_lines = draw_epipolar_lines(img2, lines2, inliers2, (255, 0, 0))  # Blue lines

# Visualize the images with epipolar lines
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img1_lines, cv2.COLOR_BGR2RGB))
plt.title('First Image with Epipolar Lines')
plt.axis('off')
plt.subplot(122)
plt.imshow(cv2.cvtColor(img2_lines, cv2.COLOR_BGR2RGB))
plt.title('Second Image with Epipolar Lines')
plt.axis('off')
plt.show()

# Step 5: Triangulate 3D points
# For triangulation, we need camera matrices P and P'
# First, we need to recover P and P' from F (requires camera intrinsics)
# For simplicity, assume intrinsics K and K' (in practice, calibrate the camera)
K = np.array([
    [500,   0, 320],
    [  0, 500, 240],
    [  0,   0,   1]
], dtype=np.float32)
K_prime = K  # Assume same intrinsics for both cameras (simplification)

# Compute essential matrix E from F: E = K'^T F K
E = K.T @ F @ K

# Decompose E to get R and t
_, R, t, _ = cv2.recoverPose(E, inliers1, inliers2, K)

# Construct camera matrices
P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])  # First camera: [I | 0]
P2 = K @ np.hstack([R, t])  # Second camera: [R | t]

# Triangulate points
points_4d = cv2.triangulatePoints(P1, P2, inliers1.T, inliers2.T)
points_3d = points_4d[:3] / points_4d[3]  # Convert from homogeneous to 3D coordinates
points_3d = points_3d.T

print("\nTriangulated 3D Points:")
print(points_3d)

# Step 6: Visualize the 3D points (optional)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Triangulated 3D Points')
plt.show()