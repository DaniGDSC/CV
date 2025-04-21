import numpy as np
import cv2

print("=== Epipole Computation ===")

# Step 1: Define synthetic camera parameters and generate 2D correspondences
# Define camera intrinsics and extrinsics
K = np.array([
    [500,   0, 320],
    [  0, 500, 240],
    [  0,   0,   1]
], dtype=np.float32)
K_prime = np.array([
    [600,   0, 300],
    [  0, 600, 200],
    [  0,   0,   1]
], dtype=np.float32)

# First camera: identity rotation, zero translation
R1 = np.eye(3)
t1 = np.array([0, 0, 0])
P = K @ np.hstack([R1, t1.reshape(3, 1)])

# Second camera: rotation around Z-axis by 30 degrees, translation
theta = np.radians(30)
R2 = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [           0,             0, 1]
])
t2 = np.array([1, 0, 5])
P_prime = K_prime @ np.hstack([R2, t2.reshape(3, 1)])

# Generate 3D points and project them to get 2D correspondences
points_3d = np.array([
    [1, 1, 5, 1],
    [2, 2, 6, 1],
    [3, 1, 5, 1],
    [1, 3, 5, 1],
    [2, 1, 7, 1],
    [1, 2, 6, 1],
    [3, 3, 5, 1],
    [2, 3, 6, 1]
])

points_1 = []  # 2D points in first image
points_2 = []  # 2D points in second image
for X in points_3d:
    x = P @ X
    x = x / x[2]
    points_1.append(x[:2])  # Take only x, y (not the homogeneous coordinate)
    
    x_prime = P_prime @ X
    x_prime = x_prime / x_prime[2]
    points_2.append(x_prime[:2])

points_1 = np.array(points_1, dtype=np.float32)
points_2 = np.array(points_2, dtype=np.float32)

# Step 2: Compute the fundamental matrix F using OpenCV
F, mask = cv2.findFundamentalMat(points_1, points_2, cv2.FM_8POINT)

# Filter inliers (optional, for verification)
inliers1 = points_1[mask.ravel() == 1]
inliers2 = points_2[mask.ravel() == 1]

# Step 3: Compute the epipole e' (right null space of F)
# Compute the SVD of F
U_F, S_F, Vt_F = np.linalg.svd(F)
# The epipole e' is the last column of V (corresponding to the smallest singular value)
e_prime = Vt_F[-1]

# Convert to 2D coordinates (optional, for interpretation)
if abs(e_prime[2]) > 1e-10:  # Avoid division by zero
    e_prime_2d = e_prime[:2] / e_prime[2]
else:
    e_prime_2d = np.array([np.nan, np.nan])  # Epipole at infinity

# Step 4: Compute the epipole e (left null space of F, for completeness)
# The epipole e is the last column of U (since F e = 0)
e = U_F[:, -1]

# Convert to 2D coordinates (optional)
if abs(e[2]) > 1e-10:
    e_2d = e[:2] / e[2]
else:
    e_2d = np.array([np.nan, np.nan])

# Print results
print("Fundamental Matrix F (computed with OpenCV):")
print(F)
print("\nEpipole e' (in second image, homogeneous coordinates):")
print(e_prime)
print("\nEpipole e' (in second image, 2D coordinates):")
print(e_prime_2d)
print("\nEpipole e (in first image, homogeneous coordinates):")
print(e)
print("\nEpipole e (in first image, 2D coordinates):")
print(e_2d)

# Step 5: Verify that F^T e' = 0 and F e = 0
print("\nVerification:")
print("F^T e' =", F.T @ e_prime)
print("F e =", F @ e)

# Step 6: Verify the epipolar constraint x'^T F x = 0 for inlier points
print("\nEpipolar Constraint Verification:")
for i in range(len(inliers1)):
    x = np.append(inliers1[i], 1)  # Convert back to homogeneous coordinates
    x_prime = np.append(inliers2[i], 1)
    constraint = x_prime.T @ F @ x
    print(f"Point pair {i+1}: {constraint:.6f}")