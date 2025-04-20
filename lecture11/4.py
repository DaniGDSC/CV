import numpy as np

# --- Exercise 4: Fundamental Matrix with 8-Point Algorithm ---
print("=== Fundamental Matrix with 8-Point Algorithm ===")

# Step 1: Define synthetic 2D point correspondences
# For demonstration, we'll create points by projecting 3D points with known camera matrices
# Define camera intrinsics and extrinsics
K = np.array([
    [500,   0, 320],
    [  0, 500, 240],
    [  0,   0,   1]
])
K_prime = np.array([
    [600,   0, 300],
    [  0, 600, 200],
    [  0,   0,   1]
])

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
    points_1.append(x)
    
    x_prime = P_prime @ X
    x_prime = x_prime / x_prime[2]
    points_2.append(x_prime)

points_1 = np.array(points_1)
points_2 = np.array(points_2)

# Step 2: Construct the A matrix
A = np.zeros((len(points_1), 9))
for i in range(len(points_1)):
    x_m, y_m, _ = points_1[i]
    x_m_prime, y_m_prime, _ = points_2[i]
    A[i] = [
        x_m_prime * x_m, x_m_prime * y_m, x_m_prime,
        y_m_prime * x_m, y_m_prime * y_m, y_m_prime,
        x_m, y_m, 1
    ]

# Step 3: Solve A f = 0 using SVD
U, S, Vt = np.linalg.svd(A)
f = Vt[-1]  # Last column of V
F = f.reshape(3, 3)

# Step 4: Enforce rank-2 constraint on F
U_F, S_F, Vt_F = np.linalg.svd(F)
S_F = np.diag([S_F[0], S_F[1], 0])  # Set smallest singular value to 0
F = U_F @ S_F @ Vt_F

# Normalize F for better readability (optional)
F = F / np.linalg.norm(F)

# Print results
print("2D Points in First Image (x_m):")
print(points_1)
print("\n2D Points in Second Image (x'_m):")
print(points_2)
print("\nEstimated Fundamental Matrix F (after rank-2 enforcement):")
print(F)

# Step 5: Verify the epipolar constraint x'^T F x = 0
print("\nVerification of Epipolar Constraint:")
for i in range(len(points_1)):
    x = points_1[i]
    x_prime = points_2[i]
    constraint = x_prime.T @ F @ x
    print(f"Point pair {i+1}: {constraint:.6f}")