import numpy as np

# --- Exercise 3: Essential Matrix Estimation ---
print("=== Essential Matrix Estimation ===")

# Step 1: Define synthetic data (intrinsics K, K', and 2D point correspondences)
# Intrinsic matrices K and K'
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

# Synthetic 2D point correspondences (at least 8 points needed for 8-point algorithm)
# For demonstration, we'll create points by projecting a 3D point with known camera matrices
# Define two camera matrices P and P' to generate synthetic correspondences
R1 = np.eye(3)
t1 = np.array([0, 0, 0])
P = K @ np.hstack([R1, t1.reshape(3, 1)])

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

# Step 2: Normalize the points using intrinsics
points_1_norm = []
points_2_norm = []
K_inv = np.linalg.inv(K)
K_prime_inv = np.linalg.inv(K_prime)

for x in points_1:
    x_norm = K_inv @ x
    points_1_norm.append(x_norm)

for x_prime in points_2:
    x_prime_norm = K_prime_inv @ x_prime
    points_2_norm.append(x_prime_norm)

points_1_norm = np.array(points_1_norm)
points_2_norm = np.array(points_2_norm)

# Step 3: Apply the 8-point algorithm to compute F
# Build the A matrix
A = np.zeros((len(points_1_norm), 9))
for i in range(len(points_1_norm)):
    u, v, _ = points_1_norm[i]
    u_prime, v_prime, _ = points_2_norm[i]
    A[i] = [
        u_prime * u, u_prime * v, u_prime,
        v_prime * u, v_prime * v, v_prime,
        u, v, 1
    ]

# Solve A f = 0 using SVD
U, S, Vt = np.linalg.svd(A)
f = Vt[-1]  # Last column of V
F = f.reshape(3, 3)

# Enforce rank-2 constraint on F
U_F, S_F, Vt_F = np.linalg.svd(F)
S_F = np.diag([S_F[0], S_F[1], 0])  # Set smallest singular value to 0
F = U_F @ S_F @ Vt_F

# Step 4: Compute E = K'^T F K
E = K_prime.T @ F @ K

# Normalize E for better readability (optional)
E = E / np.linalg.norm(E)

# Print results
print("Intrinsic Matrix K:")
print(K)
print("\nIntrinsic Matrix K':")
print(K_prime)
print("\nFundamental Matrix F (after rank-2 enforcement):")
print(F)
print("\nEstimated Essential Matrix E:")
print(E)

# Step 5: Verify the epipolar constraint x'^T E x = 0
print("\nVerification of Epipolar Constraint (should be close to 0):")
for i in range(len(points_1)):
    x = points_1[i]
    x_prime = points_2[i]
    constraint = x_prime.T @ E @ x
    print(f"Point pair {i+1}: {constraint:.6f}")