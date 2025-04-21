import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image_path = 'img/dog.jpg'  # Thay 'lena.jpg' bằng đường dẫn ảnh của bạn
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Không thể đọc ảnh tại đường dẫn: {image_path}. Kiểm tra đường dẫn hoặc tệp!")

# Lấy kích thước ảnh
h, w = image.shape[:2]

# Xác định hệ số cắt (shear factor)
s = 0.5  # Hệ số cắt dọc theo trục x

# Tạo ma trận cắt 2x3 cho cv2.warpAffine
shear_matrix = np.float32([[1, s, 0],
                           [0, 1, 0]])

# Tính kích thước mới sau khi cắt
new_w = w + int(h * s)  # Chiều rộng mới = chiều rộng gốc + chiều cao * hệ số cắt

# Thực hiện biến đổi cắt với warpAffine
sheared_image = cv2.warpAffine(image, shear_matrix, (new_w, h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# Hiển thị ảnh gốc và ảnh đã cắt
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Ảnh gốc
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

# Ảnh đã cắt
axes[1].imshow(cv2.cvtColor(sheared_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Sheared Image (Shear Factor 0.5 along x-axis)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
