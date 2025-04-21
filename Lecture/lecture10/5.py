import numpy as np

# Step 1: Define a synthetic camera matrix P (or use the P from Exercise 4)
# For demonstration, let's create a synthetic P = K [R | t]
# Intrinsic matrix K (focal lengths fx = 500, fy = 500, principal point (cx, cy) = (320, 240))
K = np.array([
    [500,   0, 320],
    [  0, 500, 240],
    [  0,   0,   1]
])

# Rotation matrix R (a simple rotation around the Z-axis by 30 degrees)
theta = np.radians(30)
R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [           0,             0, 1]
])

# Translation vector t
t = np.array([1, 2, 3])

# Construct P = K [R | t]
Rt = np.hstack([R, t.reshape(3, 1)])  # [R | t]
P = K @ Rt
print("Synthetic Camera Matrix P:")
print(P)

# Step 2: Decompose P into K, R, and t
# Extract M and the last column
M = P[:, 0:3]  # First 3 columns
p4 = P[:, 3]   # Last column

# Compute t = -M^(-1) * p4
M_inv = np.linalg.inv(M)
t = -M_inv @ p4
print("\nTranslation Vector t:")
print(t)

# Step 3: Decompose M^(-1) using QR decomposition
# M = K R, so M^(-1) = R^(-1) K^(-1)
Q, R = np.linalg.qr(M_inv)

# From QR decomposition of M^(-1):
# Q = R^(-1), so R = Q^T
# R = K^(-1), so K = R^(-1)
R_decomp = Q.T  # Rotation matrix
K_decomp = np.linalg.inv(R)  # Intrinsic matrix

# Step 4: Ensure K has positive diagonal elements
# If K has negative diagonal elements, adjust K and R
D = np.diag(np.sign(np.diag(K_decomp)))  # Diagonal matrix with +1 or -1
K_decomp = K_decomp @ D
R_decomp = D @ R_decomp

print("\nIntrinsic Matrix K:")
print(K_decomp)

print("\nRotation Matrix R:")
print(R_decomp)

# Step 5: Verify the decomposition by reconstructing P
P_reconstructed = K_decomp @ np.hstack([R_decomp, t.reshape(3, 1)])
print("\nReconstructed Camera Matrix P:")
print(P_reconstructed)

print("\nDifference between Original and Reconstructed P:")
print(np.abs(P - P_reconstructed))