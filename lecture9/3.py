import numpy as np

# Hàm chiếu điểm 3D bằng ma trận camera P
def full_camera_projection(Xw, Yw, Zw, K, R, t):
    """
    Chiếu một điểm 3D (Xw, Yw, Zw) trong không gian thế giới thành tọa độ 2D (x, y) trên ảnh.
    
    Parameters:
    - Xw, Yw, Zw: Tọa độ 3D trong không gian thế giới
    - K: Ma trận nội tại (3x3)
    - R: Ma trận quay (3x3)
    - t: Vector tịnh tiến (3x1)
    
    Returns:
    - (x, y): Tọa độ 2D trên mặt phẳng ảnh
    """
    # Tạo ma trận ngoại tại [R|t]
    Rt = np.hstack((R, t.reshape(3, 1)))  # [R|t] là ma trận 3x4
    
    # Tạo ma trận camera P = K[R|t]
    P = K @ Rt  # P là ma trận 3x4
    
    # Tọa độ 3D trong không gian thế giới ở dạng đồng nhất
    point_3d = np.array([Xw, Yw, Zw, 1])
    
    # Chiếu điểm 3D bằng ma trận P
    point_2d_homogeneous = P @ point_3d
    
    # Chuẩn hóa tọa độ đồng nhất
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
    px, py = 0, 0  # Điểm chính
    K = np.array([
        [f, 0, px],
        [0, f, py],
        [0, 0, 1]
    ])
    
    # Ma trận quay R (quay đơn vị - không quay)
    R = np.eye(3)
    
    # Vector tịnh tiến t (dịch chuyển camera)
    t = np.array([0, 0, 0])
    
    # Điểm 3D trong không gian thế giới
    Xw, Yw, Zw = 1, 1, 2
    
    try:
        # Thực hiện phép chiếu
        x, y = full_camera_projection(Xw, Yw, Zw, K, R, t)
        
        # In kết quả
        print("Bài tập 3: Extrinsic Parameters and Full Camera Matrix")
        print(f"Điểm 3D: ({Xw}, {Yw}, {Zw})")
        print(f"Ma trận nội tại K:\n{K}")
        print(f"Ma trận quay R:\n{R}")
        print(f"Vector tịnh tiến t: {t}")
        print(f"Tọa độ 2D: ({x}, {y})")
    
    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")

if __name__ == "__main__":
    main()