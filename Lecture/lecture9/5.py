import numpy as np
import cv2
import matplotlib.pyplot as plt

# Hàm chiếu điểm 3D và vẽ lên ảnh
def project_and_draw_points(image_path, P, points_3d):
    """
    Chiếu các điểm 3D lên ảnh và đánh dấu các điểm chiếu bằng vòng tròn.
    
    Parameters:
    - image_path: Đường dẫn đến ảnh
    - P: Ma trận camera (3x4)
    - points_3d: Danh sách các điểm 3D [(X, Y, Z), ...]
    
    Returns:
    - image: Ảnh với các điểm chiếu được đánh dấu
    - projected_points: Danh sách các tọa độ 2D được chiếu
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không thể đọc ảnh. Kiểm tra đường dẫn!")
    
    # Danh sách để lưu các tọa độ 2D được chiếu
    projected_points = []
    
    # Chiếu từng điểm 3D và vẽ lên ảnh
    for X, Y, Z in points_3d:
        # Tọa độ 3D ở dạng đồng nhất
        point_3d = np.array([X, Y, Z, 1])
        
        # Chiếu điểm 3D bằng ma trận P
        point_2d_homogeneous = P @ point_3d
        
        # Chuẩn hóa tọa độ đồng nhất
        w = point_2d_homogeneous[2]
        if w == 0:
            print(f"Điểm ({X}, {Y}, {Z}) có w = 0, bỏ qua.")
            continue
        
        x = int(point_2d_homogeneous[0] / w)
        y = int(point_2d_homogeneous[1] / w)
        
        # Kiểm tra xem điểm có nằm trong ảnh không
        h, w = image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            # Vẽ vòng tròn tại tọa độ (x, y)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Màu đỏ, bán kính 5
            projected_points.append((x, y))
        else:
            print(f"Điểm ({X}, {Y}, {Z}) chiếu ra ngoài ảnh: ({x}, {y})")
    
    return image, projected_points

# Hàm chính để chạy chương trình
def main():
    # Định nghĩa ma trận camera P (giả định)
    # P = K[R|t], với K là ma trận nội tại, [R|t] là ma trận ngoại tại
    f = 1000  # Tiêu cự (giả định)
    px, py = 320, 240  # Điểm chính (giả định ảnh 640x480)
    K = np.array([
        [f, 0, px],
        [0, f, py],
        [0, 0, 1]
    ])
    R = np.eye(3)  # Ma trận quay (không quay)
    t = np.array([0, 0, 5])  # Vector tịnh tiến (camera cách 5 đơn vị)
    Rt = np.hstack((R, t.reshape(3, 1)))
    P = K @ Rt  # Ma trận camera P
    
    # Tạo tập hợp các điểm 3D (giả định một lưới 3D nhỏ trên mặt phẳng checkerboard)
    points_3d = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            X = i * 0.1  # Khoảng cách nhỏ để các điểm nằm trong ảnh
            Y = j * 0.1
            Z = 5  # Z cố định (cùng với t)
            points_3d.append((X, Y, Z))
    
    # Đường dẫn đến ảnh (thay bằng ảnh của bạn)
    image_path = 'img/chessboard.jpg'
    
    try:
        # Chiếu và vẽ các điểm lên ảnh
        result_image, projected_points = project_and_draw_points(image_path, P, points_3d)
        
        # Hiển thị ảnh kết quả
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Projected 3D Points on Image')
        plt.axis('off')
        plt.show()
        
        # In các tọa độ 2D được chiếu
        print("Các tọa độ 2D được chiếu:")
        for idx, (x, y) in enumerate(projected_points):
            print(f"Điểm {idx + 1}: ({x}, {y})")
    
    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")

if __name__ == "__main__":
    main()