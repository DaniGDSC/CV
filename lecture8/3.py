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

# Tạo đối tượng SIFT
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
    if m.distance < 0.75 * n.distance:  # Tỷ lệ 0.75 (có thể điều chỉnh)
        good_matches.append(m)

# Sắp xếp các cặp ghép nối theo khoảng cách (tốt nhất trước)
good_matches = sorted(good_matches, key=lambda x: x.distance)

# Giới hạn số lượng cặp ghép nối để hiển thị (ví dụ: 50 cặp tốt nhất)
max_matches_to_show = 50
good_matches = good_matches[:min(max_matches_to_show, len(good_matches))]

# Vẽ các cặp ghép nối
matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Hiển thị kết quả
plt.figure(figsize=(15, 5))
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title('SIFT Feature Matching')
plt.axis('off')
plt.show()

# In số lượng cặp ghép nối
print(f"Số lượng cặp ghép nối tốt: {len(good_matches)}")