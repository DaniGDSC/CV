import numpy as np

# Hàm chiếu điểm 3D bằng ma trận nội tại K
def intrinsic_projection(X, Y, Z, f, px, py):
    """
    Chiếu một điểm 3D (X, Y, Z) thành tọa độ 2D (x, y) bằng ma trận nội tại K.
    
    Parameters:
    - X, Y, Z: Tọa độ 3D của điểm
    - f: Tiêu cự (focal length)
    - px, py: Tọa độ điểm chính (principal point)
    
    Returns:
    - (x, y): Tọa độ 2D trên mặt phẳng ảnh
    """
    # Xây dựng ma trận nội tại K
    K = np.array([
        [f, 0, px],
        [0, f, py],
        [0, 0, 1]
    ])
    
    # Tọa độ 3D ở dạng vector
    point_3d = np.array([X, Y, Z])
    
    # Chiếu điểm 3D bằng ma trận K
    point_2d_homogeneous = K @ point_3d
    
    # Chuẩn hóa tọa độ đồng nhất (chia cho w)
    w = point_2d_homogeneous[2]
    if w == 0:
        raise ValueError("w không được bằng 0 khi chuẩn hóa tọa độ đồng nhất!")
    
    x = point_2d_homogeneous[0] / w
    y = point_2d_homogeneous[1] / w
    return x, y

# Hàm chính để chạy chương trình
def main():
    # Định nghĩa các tham số
    f = 1  # Tiêu cự
    px, py = 0, 0  # Điểm chính (giả sử ở gốc tọa độ)
    X, Y, Z = 1, 1, 2  # Điểm 3D ví dụ
    
    try:
        # Thực hiện phép chiếu
        x, y = intrinsic_projection(X, Y, Z, f, px, py)
        
        # In kết quả
        print("Bài tập 2: Intrinsic Camera Matrix")
        print(f"Điểm 3D: ({X}, {Y}, {Z})")
        print(f"Tiêu cự f: {f}")
        print(f"Điểm chính (px, py): ({px}, {py})")
        print(f"Tọa độ 2D: ({x}, {y})")
    
    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")

if __name__ == "__main__":
    main()