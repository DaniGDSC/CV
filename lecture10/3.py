import numpy as np
import matplotlib.pyplot as plt

# Hàm thực hiện phép chiếu phối cảnh
def perspective_projection(X, Y, Z, f):
    """
    Chiếu một điểm 3D (X, Y, Z) thành tọa độ 2D (x, y) bằng phép chiếu phối cảnh.
    
    Parameters:
    - X, Y, Z: Tọa độ 3D của điểm
    - f: Tiêu cự (focal length)
    
    Returns:
    - (x, y): Tọa độ 2D trên mặt phẳng ảnh
    """
    if Z == 0:
        raise ValueError("Z không được bằng 0 trong phép chiếu phối cảnh!")
    
    x = f * X / Z
    y = f * Y / Z
    return x, y

# Hàm thực hiện phép chiếu trực giao
def orthographic_projection(X, Y, Z):
    """
    Chiếu một điểm 3D (X, Y, Z) thành tọa độ 2D (x, y) bằng phép chiếu trực giao.
    
    Parameters:
    - X, Y, Z: Tọa độ 3D của điểm
    
    Returns:
    - (x, y): Tọa độ 2D trên mặt phẳng ảnh
    """
    # Ma trận chiếu trực giao (bỏ qua Z)
    P = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    # Tọa độ 3D ở dạng đồng nhất
    point_3d = np.array([X, Y, Z, 1])
    
    # Chiếu điểm 3D bằng ma trận P
    point_2d_homogeneous = P @ point_3d
    
    # Chuẩn hóa tọa độ đồng nhất
    w = point_2d_homogeneous[2]
    if w == 0:
        raise ValueError("w không được bằng 0 khi chuẩn hóa tọa độ đồng nhất!")
    
    x = point_2d_homogeneous[0] / w
    y = point_2d_homogeneous[1] / w
    return x, y

# Hàm trực quan hóa so sánh giữa hai phép chiếu
def visualize_projection_comparison():
    """
    Trực quan hóa so sánh giữa phép chiếu trực giao và phép chiếu phối cảnh.
    """
    # Tạo tập hợp các điểm 3D với độ sâu thay đổi
    grid_size = 5
    points_3d = []
    for i in range(grid_size):
        for j in range(grid_size):
            X = i - (grid_size - 1) / 2  # X từ -2 đến 2
            Y = j - (grid_size - 1) / 2  # Y từ -2 đến 2
            Z = 2 + (i + j) * 0.2  # Z thay đổi từ 2 đến 4
            points_3d.append((X, Y, Z))
    
    # Tiêu cự f cố định cho phép chiếu phối cảnh
    f = 1
    
    # Chiếu các điểm bằng cả hai phương pháp
    orthographic_points = []
    perspective_points = []
    
    for X, Y, Z in points_3d:
        try:
            # Phép chiếu trực giao
            x_o, y_o = orthographic_projection(X, Y, Z)
            orthographic_points.append((x_o, y_o))
            
            # Phép chiếu phối cảnh
            x_p, y_p = perspective_projection(X, Y, Z, f)
            perspective_points.append((x_p, y_p))
        
        except Exception as e:
            print(f"Đã xảy ra lỗi: {str(e)}")
            continue
    
    # Chuyển danh sách điểm thành mảng để vẽ
    orthographic_points = np.array(orthographic_points)
    perspective_points = np.array(perspective_points)
    
    # Vẽ kết quả cạnh nhau
    plt.figure(figsize=(10, 5))
    
    # Phép chiếu trực giao
    plt.subplot(1, 2, 1)
    plt.scatter(orthographic_points[:, 0], orthographic_points[:, 1], c='blue', s=50)
    plt.title('Orthographic Projection')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    
    # Phép chiếu phối cảnh
    plt.subplot(1, 2, 2)
    plt.scatter(perspective_points[:, 0], perspective_points[:, 1], c='red', s=50)
    plt.title('Perspective Projection\n(f = {})'.format(f))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

# Hàm chính để chạy chương trình
def main():
    print("Bài tập 3: Orthographic Projection Simulation")
    visualize_projection_comparison()

if __name__ == "__main__":
    main()