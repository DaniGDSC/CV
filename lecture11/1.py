import numpy as np

# --- Exercise 1: Triangulation with Backprojection ---
print("=== Triangulation with Backprojection ===")

# Step 1: Define synthetic camera matrices P and P' (3x4 matrices)
# For demonstration, let's create P and P' using known intrinsics, rotation, and translation
K = np.array([
    [500,   0, 320],
    [  0, 500, 240],
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
P_prime = K @ np.hstack([R2, t2.reshape(3, 1)])

# Step 2: Define a 3D point X and project it to get 2D points x and x'
X_true = np.array([2, 3, 5, 1])  # True 3D point in homogeneous coordinates
x = P @ X_true  # Project with P
x = x / x[2]    # Normalize to get 2D point [x_1, x_2, 1]
x_prime = P_prime @ X_true  # Project with P'
x_prime = x_prime / x_prime[2]  # Normalize

# Print the camera matrices and 2D points
print("Camera Matrix P:")
print(P)
print("\nCamera Matrix P':")
print(P_prime)
print("\n2D Point x (from P):")
print(x)
print("\n2D Point x' (from P'):")
print(x_prime)

# Step 3: Build the A matrix using cross product equations
# For x and P: x × (P X) = 0 gives two equations
# For x' and P': x' × (P' X) = 0 gives two more equations
def build_A_matrix(x, P, x_prime, P_prime):
    # Extract rows of P and P'
    p1, p2, p3 = P[0], P[1], P[2]
    p1_prime, p2_prime, p3_prime = P_prime[0], P_prime[1], P_prime[2]
    
    # 2D points
    x1, x2 = x[0], x[1]
    x1_prime, x2_prime = x_prime[0], x_prime[1]
    
    # Build the A matrix (4x4)
    A = np.zeros((4, 4))
    # From x × (P X) = 0
    A[0] = x2 * p3 - p2  # x2 (p3^T X) - (p2^T X) = 0
    A[1] = p1 - x1 * p3  # (p1^T X) - x1 (p3^T X) = 0
    # From x' × (P' X) = 0
    A[2] = x2_prime * p3_prime - p2_prime
    A[3] = p1_prime - x1_prime * p3_prime
    
    return A

# Build the A matrix
A = build_A_matrix(x, P, x_prime, P_prime)
print("\nA Matrix:")
print(A)

# Step 4: Solve A X = 0 using SVD
U, S, Vt = np.linalg.svd(A)
# The solution X is the last column of V (corresponding to the smallest singular value)
X = Vt[-1]
# Convert from homogeneous coordinates to 3D
X = X / X[3]  # Normalize by the last coordinate
X_3d = X[:3]  # Take the first 3 coordinates

# Print the result
print("\nEstimated 3D Point X (homogeneous):")
print(X)
print("\nEstimated 3D Point X (Cartesian):")
print(X_3d)
print("\nTrue 3D Point (for comparison):")
print(X_true[:3])

# Step 5: Verify by reprojecting X
x_reproj = P @ X
x_reproj = x_reproj / x_reproj[2]
x_prime_reproj = P_prime @ X
x_prime_reproj = x_prime_reproj / x_prime_reproj[2]

print("\nReprojected 2D Point x:")
print(x_reproj)
print("Original 2D Point x:")
print(x)
print("\nReprojected 2D Point x':")
print(x_prime_reproj)
print("Original 2D Point x':")
print(x_prime)