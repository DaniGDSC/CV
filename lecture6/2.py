import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm tạo bộ lọc Gabor
def create_gabor_filters(n_scales=4, n_orientations=4):
    filters = []
    ksize = 31  # Kích thước kernel Gabor
    for scale in range(n_scales):
        freq = 0.05 + scale * 0.05  # Tần số tăng dần
        for theta in range(n_orientations):
            theta = theta * np.pi / n_orientations  # Góc hướng (0, 45, 90, 135 độ)
            # Tạo bộ lọc Gabor
            gabor = cv2.getGaborKernel((ksize, ksize), sigma=2.0, theta=theta, 
                                      lambd=1.0/freq, gamma=0.5, psi=0)
            filters.append(gabor)
    return filters

# Hàm tính năng lượng trung bình trên lưới 4x4
def compute_gist_descriptor(image, filters, grid_size=(4, 4)):
    # Chuyển ảnh sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    
    # Chia ảnh thành lưới 4x4
    cell_h, cell_w = h // grid_size[0], w // grid_size[1]
    descriptor = []
    
    # Lọc ảnh với từng bộ lọc Gabor
    responses = []
    for gabor in filters:
        response = cv2.filter2D(gray, cv2.CV_32F, gabor)
        responses.append(response)
    
    # Tính năng lượng trung bình trong mỗi ô
    for i in range(grid_size[0]):  # Hàng
        for j in range(grid_size[1]):  # Cột
            cell_response = []
            y_start = i * cell_h
            y_end = (i + 1) * cell_h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w
            
            for response in responses:
                # Lấy vùng tương ứng với ô
                cell = response[y_start:y_end, x_start:x_end]
                # Tính năng lượng trung bình (bình phương và lấy trung bình)
                energy = np.mean(cell ** 2)
                cell_response.append(energy)
            
            descriptor.extend(cell_response)
    
    return np.array(descriptor)

# Đọc ảnh
image = cv2.imread('img/dog.jpg')  # Thay 'image.jpg' bằng đường dẫn ảnh của bạn
if image is None:
    raise ValueError("Không thể đọc ảnh. Kiểm tra đường dẫn!")

# Tạo bộ lọc Gabor (4 tần số, 4 hướng)
filters = create_gabor_filters(n_scales=4, n_orientations=4)

# Tính GIST descriptor
gist_descriptor = compute_gist_descriptor(image, filters)

# Hiển thị ảnh gốc
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()

# In descriptor
print("GIST Descriptor:")
print(gist_descriptor)
print("Kích thước descriptor:", gist_descriptor.shape)

# Vẽ biểu đồ năng lượng trung bình
plt.figure(figsize=(10, 4))
plt.bar(range(len(gist_descriptor)), gist_descriptor)
plt.title('GIST Descriptor (Energy per Filter per Cell)')
plt.xlabel('Feature Index')
plt.ylabel('Average Energy')
plt.show()