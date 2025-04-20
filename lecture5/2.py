import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm để tính eigenvalues và eigenvectors, phân loại vùng
def analyze_corners(gray, window_size=5, k=0.04):
    h, w = gray.shape
    # Khởi tạo ma trận kết quả để lưu nhãn vùng (0: flat, 1: edge, 2: corner)
    labels = np.zeros((h, w))
    
    # Lặp qua từng pixel
    for y in range(window_size, h - window_size):
        for x in range(window_size, w - window_size):
            # Cắt vùng lân cận
            region = gray[y-window_size:y+window_size+1, x-window_size:x+window_size+1]
            
            # Tính gradient trong vùng lân cận
            dx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
            
            # Tính các sản phẩm gradient
            Ix2 = cv2.GaussianBlur(dx**2, (5, 5), 0)
            Iy2 = cv2.GaussianBlur(dy**2, (5, 5), 0)
            Ixy = cv2.GaussianBlur(dx*dy, (5, 5), 0)
            
            # Lấy giá trị trung tâm của vùng lân cận
            Ix2 = Ix2[window_size, window_size]
            Iy2 = Iy2[window_size, window_size]
            Ixy = Ixy[window_size, window_size]
            
            # Tạo ma trận M (ma trận hợp phương sai)
            M = np.array([[Ix2, Ixy],
                          [Ixy, Iy2]])
            
            # Tính eigenvalues của ma trận M
            eigenvalues = np.linalg.eigvals(M)
            lambda1, lambda2 = eigenvalues  # Sắp xếp giảm dần (giả định)
            if lambda1 < lambda2:
                lambda1, lambda2 = lambda2, lambda1
            
            # Phân loại vùng dựa trên eigenvalues
            if lambda1 < 0.01 and lambda2 < 0.01:  # Ngưỡng nhỏ, điều chỉnh theo ảnh
                labels[y, x] = 0  # Flat
            elif lambda1 > 0.01 and lambda2 < 0.01:  # Một giá trị lớn, một nhỏ
                labels[y, x] = 1  # Edge
            elif lambda1 > 0.01 and lambda2 > 0.01:  # Cả hai lớn
                labels[y, x] = 2  # Corner
            
    return labels

# Đọc ảnh
image = cv2.imread('img/chess.png')
blur = cv2.GaussianBlur(image, (5, 5), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Phân tích để xác định vùng
labels = analyze_corners(gray)

# Tạo ảnh kết quả với màu sắc cho từng loại vùng
output = image.copy()
h, w = labels.shape
for y in range(h):
    for x in range(w):
        if labels[y, x] == 0:  # Flat (màu xanh)
            output[y, x] = [0, 255, 0]
        elif labels[y, x] == 1:  # Edge (màu đỏ)
            output[y, x] = [0, 0, 255]
        elif labels[y, x] == 2:  # Corner (màu vàng)
            output[y, x] = [0, 255, 255]

# Hiển thị kết quả
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Region Classification (Green: Flat, Red: Edge, Yellow: Corner)')
plt.axis('off')
plt.show()

# In số lượng từng loại vùng
flat_count = np.sum(labels == 0)
edge_count = np.sum(labels == 1)
corner_count = np.sum(labels == 2)
print(f"Số lượng vùng Flat: {flat_count}")
print(f"Số lượng vùng Edge: {edge_count}")
print(f"Số lượng vùng Corner: {corner_count}")