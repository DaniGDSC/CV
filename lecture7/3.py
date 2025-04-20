import cv2
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread('img/dog.jpg')  # Thay 'lena.jpg' bằng đường dẫn ảnh của bạn
if image is None:
    raise ValueError("Không thể đọc ảnh. Kiểm tra đường dẫn!")

# Lấy kích thước ảnh
h, w = image.shape[:2]

# Xác định tỷ lệ phóng to
scale_x = 1.5  # Tỷ lệ theo hướng x
scale_y = 1.5  # Tỷ lệ theo hướng y

# Tính kích thước mới sau khi phóng to
new_w = int(w * scale_x)
new_h = int(h * scale_y)

# Thực hiện phóng to với cv2.resize
scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# Hiển thị ảnh gốc và ảnh đã phóng to
plt.figure(figsize=(12, 6))

# Ảnh gốc
plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Ảnh đã phóng to
plt.subplot(122)
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
plt.title('Scaled Image (1.5x)')
plt.axis('off')

plt.show()