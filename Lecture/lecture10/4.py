import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the 3D points (X_i, Y_i, Z_i) and 2D points (x_i, y_i)
# We need at least 6 correspondences for DLT to work (since P has 11 degrees of freedom)
points_3d = np.array([
    [1, 1, 1],  # Point 1
    [2, 2, 2],  # Point 2
    [3, 1, 1],  # Point 3
    [1, 3, 1],  # Point 4
    [2, 1, 3],  # Point 5
    [1, 2, 2]   # Point 6
])

points_2d = np.array([
    [0.5, 0.5],  # Corresponding 2D point 1
    [1.0, 1.0],  # Corresponding 2D point 2
    [1.5, 0.5],  # Corresponding 2D point 3
    [0.5, 1.5],  # Corresponding 2D point 4
    [0.8, 0.4],  # Corresponding 2D point 5
    [0.6, 0.8]   # Corresponding 2D point 6
])

# Step 2: Add homogeneous coordinate (1) to 3D points
points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # Shape: (n, 4)

# Step 3: Build the A matrix for DLT
def build_A_matrix(points_3d_h, points_2d):
    A = []
    for i in range(len(points_3d_h)):
        X = points_3d_h[i]  # [X_i, Y_i, Z_i, 1]
        x, y = points_2d[i]  # (x_i, y_i)
        
        # First row: [X_i^T, 0^T, -x_i X_i^T]
        row1 = np.concatenate([X, [0, 0, 0, 0], -x * X])
        # Second row: [0^T, X_i^T, -y_i X_i^T]
        row2 = np.concatenate([[0, 0, 0, 0], X, -y * X])
        
        A.append(row1)
        A.append(row2)
    
    return np.array(A)

# Build the A matrix
A = build_A_matrix(points_3d_h, points_2d)
print("A matrix shape:", A.shape)  # Should be (2n, 12)

# Step 4: Solve Ap = 0 using SVD
U, S, Vt = np.linalg.svd(A)
# The solution p is the last column of V (corresponding to the smallest singular value)
p = Vt[-1]  # Shape: (12,)

# Step 5: Reshape p into the 3x4 camera matrix P
P = p.reshape(3, 4)
print("\nEstimated Camera Matrix P:")
print(P)

# Step 6: Project the 3D points using the estimated P
def project_points(points_3d_h, P):
    projected_points = []
    for X in points_3d_h:
        # Project: x' = P * X
        x_proj = P @ X  # Shape: (3,)
        # Convert from homogeneous coordinates to 2D (divide by the last coordinate)
        x_proj_2d = x_proj[:2] / x_proj[2]
        projected_points.append(x_proj_2d)
    return np.array(projected_points)

# Project the 3D points
projected_2d = project_points(points_3d_h, P)

# Step 7: Print the original and projected points for comparison
print("\nOriginal 2D Points vs Projected 2D Points:")
for i in range(len(points_2d)):
    print(f"Point {i+1}: Original = {points_2d[i]}, Projected = {projected_2d[i]}")

# Step 8: Visualize the results using Matplotlib
plt.figure(figsize=(8, 6))

# Plot original 2D points (blue circles)
plt.scatter(points_2d[:, 0], points_2d[:, 1], c='blue', marker='o', label='Original 2D Points', s=100)

# Plot projected 2D points (red crosses)
plt.scatter(projected_2d[:, 0], projected_2d[:, 1], c='red', marker='x', label='Projected 2D Points', s=100)

# Connect corresponding points with lines for clarity
for i in range(len(points_2d)):
    plt.plot([points_2d[i, 0], projected_2d[i, 0]], 
             [points_2d[i, 1], projected_2d[i, 1]], 
             'k--', alpha=0.5)

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original vs Projected 2D Points (DLT)')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Equal scaling on both axes for better visualization
plt.show()