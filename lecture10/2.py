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

# Hàm thực hiện phép chiếu phối cảnh yếu
def weak_perspective_projection(X, Y, Z, f, Z_avg):
    """
    Chiếu một điểm 3D (X, Y, Z) thành tọa độ 2D (x, y) bằng phép chiếu phối cảnh yếu.
    
    Parameters:
    - X, Y, Z: Tọa độ 3D của điểm
    - f: Tiêu cự (focal length)
    - Z_avg: Độ sâu trung bình của các điểm
    
    Returns:
    - (x, y): Tọa độ 2D trên mặt phẳng ảnh
    """
    if Z_avg == 0:
        raise ValueError("Z_avg không được bằng 0 trong phép chiếu phối cảnh yếu!")
    
    x = f * X / Z_avg
    y = f * Y / Z_avg
    return x, y

# Hàm trực quan hóa so sánh giữa hai phép chiếu
def visualize_projection_comparison():
    """
    Trực quan hóa so sánh giữa phép chiếu phối cảnh và phép chiếu phối cảnh yếu.
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
    
    # Tính độ sâu trung bình Z_avg
    Z_values = [Z for _, _, Z in points_3d]
    Z_avg = np.mean(Z_values)
    
    # Tiêu cự f cố định
    f = 1
    
    # Chiếu các điểm bằng cả hai phương pháp
    perspective_points = []
    weak_perspective_points = []
    
    for X, Y, Z in points_3d:
        try:
            # Phép chiếu phối cảnh
            x_p, y_p = perspective_projection(X, Y, Z, f)
            perspective_points.append((x_p, y_p))
            
            # Phép chiếu phối cảnh yếu
            x_wp, y_wp = weak_perspective_projection(X, Y, Z, f, Z_avg)
            weak_perspective_points.append((x_wp, y_wp))
        
        except Exception as e:
            print(f"Đã xảy ra lỗi: {str(e)}")
            continue
    
    # Chuyển danh sách điểm thành mảng để vẽ
    perspective_points = np.array(perspective_points)
    weak_perspective_points = np.array(weak_perspective_points)
    
    # Vẽ kết quả cạnh nhau
    plt.figure(figsize=(10, 5))
    
    # Phép chiếu phối cảnh
    plt.subplot(1, 2, 1)
    plt.scatter(perspective_points[:, 0], perspective_points[:, 1], c='blue', s=50)
    plt.title('Perspective Projection')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    
    # Phép chiếu phối cảnh yếu
    plt.subplot(1, 2, 2)
    plt.scatter(weak_perspective_points[:, 0], weak_perspective_points[:, 1], c='red', s=50)
    plt.title('Weak Perspective Projection\n(Z_avg = {:.2f})'.format(Z_avg))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

# Hàm chính để chạy chương trình
def main():
    print("Weak Perspective vs. Perspective Projection")
    visualize_projection_comparison()

if __name__ == "__main__":
    main()