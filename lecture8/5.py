import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm trộn hai ảnh (blending)
def blend_images(img1, img2_warped):
    # Tạo mask cho vùng chồng lấn
    mask1 = np.ones_like(img1, dtype=np.float32)
    mask2 = np.zeros_like(img2_warped, dtype=np.float32)
    
    # Điền mask2 với giá trị 1 ở những vùng có pixel từ img2_warped
    mask2[img2_warped > 0] = 1
    
    # Tính alpha blending cho vùng chồng lấn
    alpha = 0.5  # Hệ số trộn (có thể điều chỉnh)
    blended = np.zeros_like(img1, dtype=np.float32)
    
    # Trộn ảnh
    for c in range(img1.shape[2]):  # Lặp qua các kênh màu
        blended[:, :, c] = img1[:, :, c] * (1 - alpha * mask2[:, :, c]) + \
                           img2_warped[:, :, c] * (alpha * mask2[:, :, c])
    
    return blended.astype(np.uint8)

# Đọc hai ảnh đầu vào
img1 = cv2.imread('img/class2.jpg')  # Thay 'image1.jpg' bằng đường dẫn ảnh thứ nhất
img2 = cv2.imread('img/class1.jpg')  # Thay 'image2.jpg' bằng đường dẫn ảnh thứ hai
if img1 is None or img2 is None:
    raise ValueError("Không thể đọc một hoặc cả hai ảnh. Kiểm tra đường dẫn!")

# Chuyển ảnh sang ảnh xám
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Tạo đối tượng SIFT để phát hiện điểm đặc trưng
sift = cv2.SIFT_create()

# Phát hiện và tính toán các điểm đặc trưng (keypoints) và descriptor
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Tạo đối tượng BFMatcher (Brute-Force Matcher) với khoảng cách L2
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Ghép nối các descriptor
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Áp dụng kiểm tra tỷ lệ (ratio test) để lọc các cặp ghép nối tốt
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # Tỷ lệ 0.75
        good_matches.append(m)

# Trích xuất tọa độ của các điểm đặc trưng tương ứng
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Ước lượng ma trận homography H bằng RANSAC
H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

# Kiểm tra xem ma trận H có được tính toán thành công không
if H is None:
    raise ValueError("Không thể ước lượng ma trận homography H. Kiểm tra lại các cặp điểm!")

# Lấy kích thước của ảnh thứ nhất và thứ hai
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# Tính kích thước của ảnh panorama
# Tọa độ các góc của ảnh thứ hai sau khi biến đổi
pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, H)
all_points = np.concatenate((dst, np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)), axis=0)

# Tính kích thước mới của ảnh panorama
x_min, y_min = np.int32(all_points.min(axis=0).ravel())
x_max, y_max = np.int32(all_points.max(axis=0).ravel())
panorama_w = x_max - x_min
panorama_h = y_max - y_min

# Tạo ma trận dịch chuyển để đưa ảnh về gốc tọa độ (0, 0)
translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
H_adjusted = translation_matrix @ H

# Biến đổi phối cảnh ảnh thứ hai
warped_img2 = cv2.warpPerspective(img2, H_adjusted, (panorama_w, panorama_h))

# Đặt ảnh thứ nhất vào ảnh panorama
panorama = np.zeros((panorama_h, panorama_w, 3), dtype=np.uint8)
panorama[-y_min:-y_min+h1, -x_min:-x_min+w1] = img1

# Trộn hai ảnh
panorama = blend_images(panorama, warped_img2)

# Hiển thị kết quả
plt.figure(figsize=(15, 5))

# Ảnh thứ nhất
plt.subplot(131)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('Image 1')
plt.axis('off')

# Ảnh thứ hai (gốc)
plt.subplot(132)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Image 2')
plt.axis('off')

# Ảnh panorama
plt.subplot(133)
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.title('Panorama')
plt.axis('off')

plt.show()

# In số lượng cặp ghép nối tốt
print(f"Số lượng cặp ghép nối tốt: {len(good_matches)}")
# In ma trận homography H
print("Ma trận homography H:")
print(H)