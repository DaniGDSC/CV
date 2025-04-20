import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread('img/dog.jpg')  # Thay 'lena.jpg' bằng đường dẫn ảnh của bạn
if image is None:
    raise ValueError("Không thể đọc ảnh. Kiểm tra đường dẫn!")

# Lấy kích thước ảnh
h, w = image.shape[:2]

# Xác định tâm xoay (trung tâm của ảnh)
center = (w // 2, h // 2)

# Xác định góc xoay (45 độ)
angle = 45

# Tạo ma trận xoay 2x3 bằng cv2.getRotationMatrix2D
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

# Tính kích thước mới để chứa toàn bộ ảnh sau khi xoay
# Khi xoay, kích thước ảnh có thể thay đổi, ta tính kích thước mới để tránh cắt mất nội dung
cos_val = np.abs(np.cos(np.radians(angle)))
sin_val = np.abs(np.sin(np.radians(angle)))
new_w = int((h * sin_val) + (w * cos_val))
new_h = int((h * cos_val) + (w * sin_val))

# Điều chỉnh ma trận xoay để đặt ảnh vào giữa khung mới
rotation_matrix[0, 2] += (new_w / 2) - center[0]
rotation_matrix[1, 2] += (new_h / 2) - center[1]

# Thực hiện xoay với warpAffine
rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# Hiển thị ảnh gốc và ảnh đã xoay
plt.figure(figsize=(12, 6))

# Ảnh gốc
plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Ảnh đã xoay
plt.subplot(122)
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title('Rotated Image (45 Degrees)')
plt.axis('off')

plt.show()