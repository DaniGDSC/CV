import numpy as np
import cv2
import matplotlib.pyplot as plt

# Hàm ước lượng vị trí camera và trực quan hóa
def estimate_camera_pose(image_path, checkerboard_size, square_size, K, dist_coeffs):
    """
    Ước lượng vị trí camera từ ảnh checkerboard và trực quan hóa các điểm chiếu lại.
    
    Parameters:
    - image_path: Đường dẫn đến ảnh checkerboard
    - checkerboard_size: Kích thước checkerboard (số ô, ví dụ: (9, 6))
    - square_size: Kích thước mỗi ô (đơn vị thực tế, ví dụ: 0.025m)
    - K: Ma trận nội tại của camera (3x3)
    - dist_coeffs: Hệ số méo (distortion coefficients)
    
    Returns:
    - rvec: Vector quay
    - tvec: Vector tịnh tiến
    - image: Ảnh với các điểm chiếu lại được đánh dấu
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không thể đọc ảnh. Kiểm tra đường dẫn!")
    
    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện các góc checkerboard
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    
    if not ret:
        raise ValueError("Không thể phát hiện các góc checkerboard!")
    
    # Tinh chỉnh tọa độ góc
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Tạo tọa độ 3D của các góc checkerboard (giả định Z = 0)
    obj_points = []
    for i in range(checkerboard_size[1]):  # Số hàng
        for j in range(checkerboard_size[0]):  # Số cột
            obj_points.append([j * square_size, i * square_size, 0])
    obj_points = np.array(obj_points, dtype=np.float32)
    
    # Tọa độ 2D của các góc
    img_points = corners.reshape(-1, 2)
    
    # Ước lượng vị trí camera bằng solvePnP
    ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs)
    if not ret:
        raise ValueError("Không thể ước lượng vị trí camera bằng solvePnP!")
    
    # Chiếu lại các điểm 3D để kiểm tra
    img_points_reprojected, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist_coeffs)
    img_points_reprojected = img_points_reprojected.reshape(-1, 2)
    
    # Vẽ các điểm chiếu lại lên ảnh
    for pt in img_points_reprojected:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Màu đỏ, bán kính 5
    
    return rvec, tvec, image

# Tiêu chí dừng cho tinh chỉnh góc
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Hàm chính để chạy chương trình
def main():
    # Định nghĩa các tham số
    checkerboard_size = (9, 6)  # Số ô (cột, hàng) của checkerboard
    square_size = 0.025  # Kích thước mỗi ô (ví dụ: 25mm = 0.025m)
    
    # Ma trận nội tại K (giả định)
    f = 1000  # Tiêu cự
    px, py = 320, 240  # Điểm chính (giả định ảnh 640x480)
    K = np.array([
        [f, 0, px],
        [0, f, py],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Hệ số méo (giả định không có méo)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    # Đường dẫn đến ảnh (thay bằng ảnh của bạn)
    image_path = 'img/checker.png'
    
    try:
        # Ước lượng vị trí camera và trực quan hóa
        rvec, tvec, result_image = estimate_camera_pose(
            image_path, checkerboard_size, square_size, K, dist_coeffs
        )
        
        # In kết quả
        print("Bài tập 6: Pose Estimation with Real Data")
        print(f"Vector quay (rvec):\n{rvec}")
        print(f"Vector tịnh tiến (tvec):\n{tvec}")
        
        # Hiển thị ảnh kết quả
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Reprojected Points on Checkerboard')
        plt.axis('off')
        plt.show()
    
    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")

if __name__ == "__main__":
    main()