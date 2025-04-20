import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm xác định ma trận biến đổi affine từ các cặp điểm bằng Least Squares
def compute_affine_transform(src_points, dst_points):
    # Đảm bảo có ít nhất 3 cặp điểm để giải hệ phương trình
    if len(src_points) != len(dst_points) or len(src_points) < 3:
        raise ValueError("Cần ít nhất 3 cặp điểm")

    # Số cặp điểm
    n = len(src_points)
    
    # Xây dựng ma trận A và vector b
    A = np.zeros((2 * n, 6))  # 2 phương trình cho mỗi cặp điểm, 6 tham số
    b = np.zeros(2 * n)
    
    for i in range(n):
        x, y = src_points[i]
        xp, yp = dst_points[i]
        
        # Phương trình cho x'
        A[2*i, 0] = x  # Hệ số a
        A[2*i, 1] = y  # Hệ số b
        A[2*i, 2] = 1  # Hệ số t_x
        b[2*i] = xp
        
        # Phương trình cho y'
        A[2*i+1, 3] = x  # Hệ số c
        A[2*i+1, 4] = y  # Hệ số d
        A[2*i+1, 5] = 1  # Hệ số t_y
        b[2*i+1] = yp
    
    # Giải hệ phương trình A*x = b bằng Least Squares
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Tạo ma trận affine 2x3 từ tham số [a, b, t_x, c, d, t_y]
    affine_matrix = np.float32([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]]
    ])
    
    return affine_matrix

# Đọc ảnh
image = cv2.imread('img/dog.jpg') 
if image is None:
    raise ValueError("Không thể đọc ảnh. Kiểm tra đường dẫn!")

# Lấy kích thước ảnh
h, w = image.shape[:2]

# Định nghĩa các cặp điểm tương ứng (src_points -> dst_points)
# Ví dụ: 4 cặp điểm (có thể thay đổi tùy ý)
src_points = np.float32([
    [0, 0],      # Góc trên-trái
    [w-1, 0],    # Góc trên-phải
    [0, h-1],    # Góc dưới-trái
    [w-1, h-1]   # Góc dưới-phải
])

dst_points = np.float32([
    [50, 50],        # Góc trên-trái dịch sang phải và xuống
    [w-50, 50],      # Góc trên-phải dịch sang trái và xuống
    [50, h-50],      # Góc dưới-trái dịch sang phải và lên
    [w-50, h-50]     # Góc dưới-phải dịch sang trái và lên
])

# Tính ma trận biến đổi affine bằng Least Squares
affine_matrix = compute_affine_transform(src_points, dst_points)

# Tính kích thước mới (dựa trên tọa độ điểm đích để chứa toàn bộ ảnh)
new_w = int(max(dst_points[:, 0]) - min(dst_points[:, 0]))
new_h = int(max(dst_points[:, 1]) - min(dst_points[:, 1]))

# Thực hiện biến đổi affine với warpAffine
transformed_image = cv2.warpAffine(image, affine_matrix, (new_w, new_h),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# Hiển thị ảnh gốc và ảnh đã biến đổi
plt.figure(figsize=(12, 6))

# Ảnh gốc
plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Ảnh đã biến đổi
plt.subplot(122)
plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
plt.title('Affine Transformed Image')
plt.axis('off')

plt.show()