import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc hai ảnh đầu vào
img1 = cv2.imread('img/class2.jpg') 
img2 = cv2.imread('img/class1.jpg')  
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

# Ước lượng ma trận đồng dạng H bằng RANSAC
H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

# Kiểm tra xem ma trận H có được tính toán thành công không
if H is None:
    raise ValueError("Không thể ước lượng ma trận đồng dạng H. Kiểm tra lại các cặp điểm!")

# Lấy kích thước của ảnh thứ nhất
h1, w1 = img1.shape[:2]

# Biến đổi phối cảnh ảnh thứ hai sang phối cảnh của ảnh thứ nhất
warped_image = cv2.warpPerspective(img2, H, (w1, h1))

# Hiển thị ảnh gốc và ảnh đã biến đổi
plt.figure(figsize=(15, 5))

# Ảnh thứ nhất
plt.subplot(131)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('Image 1')
plt.axis('off')

# Ảnh thứ hai (gốc)
plt.subplot(132)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Image 2 (Original)')
plt.axis('off')

# Ảnh thứ hai (sau khi biến đổi)
plt.subplot(133)
plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
plt.title('Image 2 (Warped to Image 1)')
plt.axis('off')

plt.show()

# In số lượng cặp ghép nối tốt
print(f"Số lượng cặp ghép nối tốt: {len(good_matches)}")
# In ma trận đồng dạng H
print("Ma trận đồng dạng H:")
print(H)