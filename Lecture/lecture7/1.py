import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread('img/dog.jpg')  # Thay 'lena.jpg' bằng đường dẫn ảnh của bạn
if image is None:
    raise ValueError("Không thể đọc ảnh. Kiểm tra đường dẫn!")

# Xác định các thông số dịch chuyển
tx = 100  # Dịch 100 pixel sang phải
ty = 50   # Dịch 50 pixel xuống dưới

# Tạo ma trận dịch chuyển 2x3 cho cv2.warpAffine
translation_matrix = np.float32([[1, 0, tx],
                                [0, 1, ty]])

# Lấy kích thước ảnh
h, w = image.shape[:2]

# Tính kích thước mới của ảnh sau khi dịch (bao gồm cả vùng đen)
new_h = h + abs(ty)  # Thêm chiều cao để chứa dịch xuống
new_w = w + abs(tx)  # Thêm chiều rộng để chứa dịch sang phải

# Thực hiện dịch chuyển với warpAffine
translated_image = cv2.warpAffine(image, translation_matrix, (new_w, new_h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# Hiển thị ảnh gốc và ảnh đã dịch
plt.figure(figsize=(12, 6))

# Ảnh gốc
plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Ảnh đã dịch
plt.subplot(122)
plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))
plt.title('Translated Image (100px Right, 50px Down)')
plt.axis('off')

plt.show()