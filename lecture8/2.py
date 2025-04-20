import cv2
import numpy as np
import argparse

def compute_homography(points1, points2):
    A = []
    for i in range(len(points1)):
        x, y = points1[i]
        x_prime, y_prime = points2[i]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape((3, 3))
    if H[2, 2] != 0:
        H = H / H[2, 2]
    else:
        raise ValueError("Homography matrix normalization failed due to division by zero.")
    return H

def warp_images(image1, image2, H):
    height, width = image1.shape[:2]
    return cv2.warpPerspective(image2, H, (width, height))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homography and Image Warping")
    parser.add_argument("img/class2.jpg", required=True, help="Path to the first image")
    parser.add_argument("img/class1.jpg", required=True, help="Path to the second image")
    args = parser.parse_args()

    image1 = cv2.imread(args.image1)
    image2 = cv2.imread(args.image2)

    if image1 is None or image2 is None:
        raise FileNotFoundError("One or both images could not be loaded. Check the file paths.")

    # Example points (replace with actual points or interactive selection)
    points1 = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
    points2 = np.array([[120, 120], [220, 120], [220, 220], [120, 220]])

    H = compute_homography(points1, points2)
    warped_image = warp_images(image1, image2, H)

    output_path = "warped_image.jpg"
    cv2.imwrite(output_path, warped_image)
    print(f"Warped image saved to {output_path}")
