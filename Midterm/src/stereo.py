import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load and resize images
def load_images(max_size=1000):
    """Load, resize, and validate stereo image pair."""
    paths = [
        r'D:\University\Computer_vision\CV\Computer_vision_assigment\Midterm\data\input\part_b\im0.png',  # Left image
        r'D:\University\Computer_vision\CV\Computer_vision_assigment\Midterm\data\input\part_b\im1.png'   # Right image
    ]
    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        h, w = img.shape[:2]
        scale = min(max_size / w, max_size / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        images.append(img)
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    
    plt.figure(figsize=(12, 3))
    for i, img in enumerate(images, 1):
        plt.subplot(1, 2, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'{"Left" if i==1 else "Right"} Image')
        plt.axis('off')
    plt.suptitle('Stereo Image Pair')
    plt.tight_layout()
    plt.savefig('input_images.png')
    plt.show()
    
    return images, gray_images

# Step 2: Compute disparity map using StereoBM
def compute_disparity_map(gray_left, gray_right):
    """Compute disparity map using basic Block Matching (StereoBM)."""
    window_size = 15  # Must be odd
    min_disp = 0
    num_disp = 64  # Multiples of 16
    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=window_size)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    # Normalize for visualization
    disparity_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(disparity_vis, cmap='jet')
    plt.colorbar(label='Disparity')
    plt.title('Disparity Map (StereoBM)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('disparity_map.png')
    plt.show()
    
    return disparity

# Step 3: Reconstruct 3D point cloud
def reconstruct_3d_point_cloud(disparity, focal_length=800, baseline=0.1):
    """Reconstruct 3D point cloud from disparity map."""
    h, w = disparity.shape
    Q = np.float32([
        [1, 0, 0, -w/2],
        [0, 1, 0, -h/2],
        [0, 0, 0, focal_length],
        [0, 0, -1/baseline, 0]
    ])
    
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    mask = (disparity > disparity.min()) & (np.abs(points_3d[:, :, 2]) < 10)
    points_3d = points_3d[mask]
    colors = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)[mask]
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=colors/255.0, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud (Projected View)')
    plt.tight_layout()
    plt.savefig('point_cloud.png')
    plt.show()
    
    return points_3d

# Step 4: Estimate fundamental matrix and draw epipolar lines
def estimate_fundamental_matrix(gray_images):
    """Estimate fundamental matrix and visualize epipolar lines."""
    sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)
    keypoints_list, descriptors_list = [], []
    for gray_img in gray_images:
        keypoints, descriptors = sift.detectAndCompute(gray_img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(descriptors_list[0], descriptors_list[1], k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"Fundamental Matrix: {len(good_matches)} good matches")
    
    pts1 = np.float32([keypoints_list[0][m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints_list[1][m.trainIdx].pt for m in good_matches])
    
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0)
    inliers = np.sum(mask)
    print(f"Fundamental Matrix: {inliers} inliers out of {len(good_matches)} matches")
    
    inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]
    
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    
    img1_lines = images[0].copy()
    img2_lines = images[1].copy()
    h, w = images[0].shape[:2]
    for i, (line1, line2) in enumerate(zip(lines1[:10], lines2[:10])):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = 0, int(-line1[2] / line1[1])
        x1, y1 = w, int(-(line1[2] + line1[0] * w) / line1[1])
        img1_lines = cv2.line(img1_lines, (x0, y0), (x1, y1), color, 1)
        x0, y0 = 0, int(-line2[2] / line2[1])
        x1, y1 = w, int(-(line2[2] + line2[0] * w) / line2[1])
        img2_lines = cv2.line(img2_lines, (x0, y0), (x1, y1), color, 1)
    
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1_lines, cv2.COLOR_BGR2RGB))
    plt.title('Epipolar Lines on Left Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2_lines, cv2.COLOR_BGR2RGB))
    plt.title('Epipolar Lines on Right Image')
    plt.axis('off')
    plt.suptitle('Epipolar Lines')
    plt.tight_layout()
    plt.savefig('epipolar_lines.png')
    plt.show()
    
    return F

# Main execution
try:
    images, gray_images = load_images(max_size=1000)
    disparity = compute_disparity_map(gray_images[0], gray_images[1])
    points_3d = reconstruct_3d_point_cloud(disparity)
    F = estimate_fundamental_matrix(gray_images)
    print("All outputs generated successfully. Check saved PNG files for report.")
except Exception as e:
    print(f"Error: {str(e)}")