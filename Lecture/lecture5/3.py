import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Hàm tính ma trận M và lỗi E(u, v)
def compute_error_surface(gray, x, y, window_size=5, k_size=3):
    # Cắt vùng lân cận quanh điểm (x, y)
    region = gray[max(0, y-window_size):min(gray.shape[0], y+window_size+1),
                  max(0, x-window_size):min(gray.shape[1], x+window_size+1)]
    
    # Tính gradient trong vùng lân cận
    dx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=k_size)
    dy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=k_size)
    
    # Tính các sản phẩm gradient và làm mịn bằng Gaussian blur
    Ix2 = cv2.GaussianBlur(dx**2, (k_size, k_size), 0)
    Iy2 = cv2.GaussianBlur(dy**2, (k_size, k_size), 0)
    Ixy = cv2.GaussianBlur(dx*dy, (k_size, k_size), 0)
    
    # Lấy giá trị trung tâm của vùng lân cận
    center_y, center_x = window_size, window_size
    Ix2 = Ix2[center_y, center_x]
    Iy2 = Iy2[center_y, center_x]
    Ixy = Ixy[center_y, center_x]
    
    # Tạo ma trận M
    M = np.array([[Ix2, Ixy],
                  [Ixy, Iy2]])
    
    # Tạo lưới giá trị [u, v]
    u = np.linspace(-5, 5, 50)
    v = np.linspace(-5, 5, 50)
    u, v = np.meshgrid(u, v)
    
    # Tính E(u, v) = [u, v] * M * [u, v].T
    E = np.zeros_like(u, dtype=float)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            displacement = np.array([u[i, j], v[i, j]])
            E[i, j] = np.dot(np.dot(displacement, M), displacement.T)
    
    return u, v, E, M

# Đọc ảnh
image = cv2.imread('img/chess.png') 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Chọn một điểm để tính (ví dụ: trung tâm ảnh)
h, w = gray.shape
x, y = w // 2, h // 2

# Tính bề mặt lỗi
u, v, E, M = compute_error_surface(gray, x, y)

# Vẽ biểu đồ 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(u, v, E, cmap='viridis')

# Thêm nhãn và tiêu đề
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_title('Error Surface at Point ({}, {})'.format(x, y))
fig.colorbar(surf)
plt.show()

# In ma trận M để tham khảo
print("Ma trận M tại điểm ({}, {}):".format(x, y))
print(M)