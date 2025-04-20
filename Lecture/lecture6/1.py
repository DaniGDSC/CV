import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Hàm để trích xuất và xử lý patch 40x40
def extract_patch(image, x, y, scale, theta, patch_size=40):
    # Chuyển đổi ảnh sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Tính kích thước patch dựa trên scale
    scaled_patch_size = int(patch_size * scale)
    if scaled_patch_size % 2 == 1:  # Đảm bảo kích thước là số chẵn
        scaled_patch_size += 1
    
    # Trích xuất patch xung quanh điểm (x, y)
    half_patch = scaled_patch_size // 2
    patch = gray[max(0, y-half_patch):min(gray.shape[0], y+half_patch),
                 max(0, x-half_patch):min(gray.shape[1], x+half_patch)]
    
    # Đảm bảo patch có kích thước đúng (scaled_patch_size x scaled_patch_size)
    if patch.shape[0] != scaled_patch_size or patch.shape[1] != scaled_patch_size:
        patch = cv2.resize(patch, (scaled_patch_size, scaled_patch_size))
    
    # Xoay patch theo góc theta (độ)
    center = (scaled_patch_size // 2, scaled_patch_size // 2)
    M = cv2.getRotationMatrix2D(center, theta, 1.0)
    patch = cv2.warpAffine(patch, M, (scaled_patch_size, scaled_patch_size))
    
    # Chuẩn hóa patch: trừ trung bình và chia cho độ lệch chuẩn
    patch -= np.mean(patch)
    std = np.std(patch)
    if std > 0:  # Tránh chia cho 0
        patch /= std
    
    return patch

# Hàm để subsample patch xuống kích thước 8x8
def subsample_patch(patch, target_size=8):
    return cv2.resize(patch, (target_size, target_size))

# Hàm để áp dụng Haar Wavelet Transform
def haar_wavelet_transform(patch):
    # Áp dụng Haar Wavelet Transform bằng cách sử dụng bộ lọc đơn giản
    coeffs = np.zeros_like(patch)
    h, w = patch.shape
    
    # Haar Wavelet Transform cơ bản (mức 1)
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            if i+1 < h and j+1 < w:
                # Tính giá trị trung bình và hiệu (LL, LH, HL, HH)
                a = patch[i, j]
                b = patch[i, j+1]
                c = patch[i+1, j]
                d = patch[i+1, j+1]
                
                # LL (trung bình)
                coeffs[i//2, j//2] = (a + b + c + d) / 4
                # LH (horizontal)
                coeffs[i//2, j//2 + w//2] = (a + c - b - d) / 4
                # HL (vertical)
                coeffs[i//2 + h//2, j//2] = (a + b - c - d) / 4
                # HH (diagonal)
                coeffs[i//2 + h//2, j//2 + w//2] = (a + d - b - c) / 4
    
    return coeffs.flatten()  # Trả về vector descriptor

# Hàm chính để tạo MOPS descriptor
def compute_mops_descriptor(image, x, y, scale, theta):
    # Bước 1: Trích xuất và chuẩn hóa patch
    patch = extract_patch(image, x, y, scale, theta)
    
    # Bước 2: Subsample patch xuống 8x8
    subsampled_patch = subsample_patch(patch, target_size=8)
    
    # Bước 3: Áp dụng Haar Wavelet Transform
    descriptor = haar_wavelet_transform(subsampled_patch)
    
    return descriptor, patch, subsampled_patch

# Đọc ảnh
image = cv2.imread('img/dog.jpg') 
if image is None:
    raise ValueError("Không thể đọc ảnh. Kiểm tra đường dẫn!")

# Định nghĩa điểm đặc trưng (x, y, scale, theta)
x, y = image.shape[1] // 2, image.shape[0] // 2  # Ví dụ: trung tâm ảnh
scale = 1.0  # Tỷ lệ scale
theta = 45.0  # Góc xoay (độ)

# Tính MOPS descriptor
descriptor, patch, subsampled_patch = compute_mops_descriptor(image, x, y, scale, theta)

# Hiển thị kết quả
plt.figure(figsize=(15, 5))

# Ảnh gốc
plt.subplot(131)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter(x, y, c='red', s=100, marker='x')  # Đánh dấu điểm đặc trưng
plt.title('Original Image')
plt.axis('off')

# Patch 40x40
plt.subplot(132)
plt.imshow(patch, cmap='gray')
plt.title('40x40 Patch (Normalized)')
plt.axis('off')

# Patch sau khi subsample (8x8)
plt.subplot(133)
plt.imshow(subsampled_patch, cmap='gray')
plt.title('Subsampled Patch (8x8)')
plt.axis('off')

plt.show()

# In descriptor
print("MOPS Descriptor (Haar Wavelet Transform):")
print(descriptor)
print("Kích thước descriptor:", descriptor.shape)