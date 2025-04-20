import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm để thực hiện Non-Maximum Suppression (NMS)
def non_max_suppression(image, response, threshold=0.1):
    corners = []
    h, w = image.shape
    response_copy = response.copy()
    
    for y in range(h):
        for x in range(w):
            if response_copy[y, x] > threshold:
                # Kiểm tra vùng lân cận (ví dụ: cửa sổ 5x5)
                window_size = 5
                y_start = max(0, y - window_size // 2)
                y_end = min(h, y + window_size // 2 + 1)
                x_start = max(0, x - window_size // 2)
                x_end = min(w, x + window_size // 2 + 1)
                
                if response_copy[y, x] == np.max(response_copy[y_start:y_end, x_start:x_end]):
                    corners.append((x, y))
                    # Đặt giá trị xung quanh về 0 để tránh trùng lặp
                    response_copy[y_start:y_end, x_start:x_end] = 0
    
    return corners

# Đọc ảnh
image = cv2.imread('img\chess.png')  # Thay 'image.jpg' bằng đường dẫn ảnh của bạn
blur = cv2.GaussianBlur(image, (5, 5), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)



# Tính gradient theo x và y
dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # Gradient theo x
dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # Gradient theo y

# Tính các sản phẩm của gradient
Ix2 = dx ** 2
Iy2 = dy ** 2
Ixy = dx * dy

# Áp dụng Gaussian blur để làm mịn
k_size = 5
Ix2 = cv2.GaussianBlur(Ix2, (k_size, k_size), 0)
Iy2 = cv2.GaussianBlur(Iy2, (k_size, k_size), 0)
Ixy = cv2.GaussianBlur(Ixy, (k_size, k_size), 0)

# Tính ma trận M và giá trị phản hồi R
k = 0.04  # Hằng số Harris
h, w = gray.shape
response = np.zeros((h, w))

for y in range(h):
    for x in range(w):
        # Ma trận M tại điểm (x, y)
        M = np.array([[Ix2[y, x], Ixy[y, x]],
                      [Ixy[y, x], Iy2[y, x]]])
        
        # Tính giá trị R (phản hồi Harris)
        det_M = np.linalg.det(M)
        trace_M = np.trace(M)
        R = det_M - k * (trace_M ** 2)
        response[y, x] = R

# Áp dụng ngưỡng để lọc các góc
threshold = 0.01 * np.max(response)
response[response < threshold] = 0

# Thực hiện Non-Maximum Suppression
corners = non_max_suppression(gray, response, threshold)

# Vẽ các góc lên ảnh gốc
for corner in corners:
    x, y = corner
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Vẽ vòng tròn xanh tại các góc

# Hiển thị kết quả
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners Detected')
plt.axis('off')
plt.show()

# In số lượng góc được phát hiện
print(f"Số lượng góc được phát hiện: {len(corners)}")
