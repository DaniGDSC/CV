import numpy as np
import matplotlib.pyplot as plt

print("=== Epipolar Line Visualization ===")

# Step 1: Define synthetic data (fundamental matrix F and 2D points in the first image)
# Synthetic fundamental matrix F (3x3)
# In a real scenario, F would be computed from image correspondences using cv2.findFundamentalMat
F = np.array([
    [ 0.0,  0.0, -0.1],
    [ 0.0,  0.0,  0.2],
    [ 0.1, -0.2,  0.0]
])

# 2D points in the first image (in homogeneous coordinates)
points_1 = np.array([
    [100, 100, 1],  # Point 1
    [200, 150, 1],  # Point 2
    [300, 200, 1]   # Point 3
])

# Step 2: Load the second image (synthetic placeholder for demonstration)
image_width, image_height = 640, 480
image2 = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255  # White image

# Step 3: Compute epipolar lines l' = F x for each point x in the first image
epipolar_lines = []
for x in points_1:
    l_prime = F @ x  # l' = F x
    epipolar_lines.append(l_prime)

# Print the epipolar lines
print("Epipolar Lines (l' = F x):")
for i, l in enumerate(epipolar_lines):
    print(f"Point {i+1}: {l}")

# Step 4: Plot the epipolar lines on the second image
plt.figure(figsize=(10, 8))
plt.imshow(image2)

# Image boundaries
x_range = np.array([0, image_width - 1])

# Plot each epipolar line
colors = ['r', 'g', 'b']  # Different colors for each line
for i, l_prime in enumerate(epipolar_lines):
    l1, l2, l3 = l_prime
    if abs(l2) > 1e-10:  # Ensure l2 is not zero to avoid division by zero
        # Line equation: l1 x' + l2 y' + l3 = 0
        # Solve for y': y' = -(l1 x' + l3) / l2
        y_range = -(l1 * x_range + l3) / l2
        plt.plot(x_range, y_range, colors[i], label=f'Epipolar Line {i+1}')
    else:
        # If l2 is zero, the line is vertical: x' = -l3 / l1
        x_val = -l3 / l1
        plt.axvline(x=x_val, color=colors[i], label=f'Epipolar Line {i+1}')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Epipolar Lines in Second Image')
plt.legend()
plt.xlim(0, image_width)
plt.ylim(image_height, 0)  # Invert y-axis to match image coordinates
plt.show()