import numpy as np
import matplotlib.pyplot as plt

# Hàm chiếu pinhole với tùy chọn thêm nhiễu
def pinhole_projection_with_noise(X, Y, Z, f, noise_level=0.0):
    """
    Chiếu một điểm 3D (X, Y, Z) thành tọa độ 2D (x, y) bằng phép chiếu pinhole,
    với tùy chọn thêm nhiễu để mô phỏng kích thước lỗ kim lớn hơn.
    
    Parameters:
    - X, Y, Z: Tọa độ 3D của điểm
    - f: Tiêu cự (focal length)
    - noise_level: Mức độ nhiễu (0.0 = không nhiễu)
    
    Returns:
    - (x, y): Tọa độ 2D trên mặt phẳng ảnh
    """
    if Z == 0:
        raise ValueError("Z không được bằng 0 trong phép chiếu pinhole!")
    
    x = f * X / Z
    y = f * Y / Z
    
    # Thêm nhiễu ngẫu nhiên để mô phỏng kích thước lỗ kim lớn hơn
    if noise_level > 0:
        x += np.random.normal(0, noise_level)
        y += np.random.normal(0, noise_level)
    
    return x, y

# Hàm trực quan hóa ảnh hưởng của tiêu cự và kích thước lỗ kim
def visualize_focal_length_and_pinhole_effect():
    """
    Trực quan hóa ảnh hưởng của tiêu cự f và kích thước lỗ kim lên phép chiếu của một lưới 3D.
    """
    # Tạo lưới 3D (5x5 điểm, Z cố định)
    grid_size = 5
    Z = 2  # Z cố định
    points_3d = []
    for i in range(grid_size):
        for j in range(grid_size):
            X = i - (grid_size - 1) / 2  # X từ -2 đến 2
            Y = j - (grid_size - 1) / 2  # Y từ -2 đến 2
            points_3d.append((X, Y, Z))
    
    # Các giá trị tiêu cự f để thử
    focal_lengths = [0.5, 1.0, 2.0]
    noise_level = 0.1  # Mức độ nhiễu để mô phỏng kích thước lỗ kim
    
    # Vẽ kết quả cho từng giá trị f
    plt.figure(figsize=(15, 5))
    for idx, f in enumerate(focal_lengths, 1):
        projected_points = []
        for X, Y, Z in points_3d:
            try:
                x, y = pinhole_projection_with_noise(X, Y, Z, f, noise_level)
                projected_points.append((x, y))
            except Exception as e:
                print(f"Đã xảy ra lỗi với f={f}: {str(e)}")
                continue
        
        # Chuyển danh sách điểm thành mảng để vẽ
        projected_points = np.array(projected_points)
        plt.subplot(1, 3, idx)
        plt.scatter(projected_points[:, 0], projected_points[:, 1], c='blue', s=50, alpha=0.6)
        plt.title(f'Focal Length f = {f}\n(Noise Level = {noise_level})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.axis('equal')  # Đảm bảo tỷ lệ trục x và y bằng nhau
    
    plt.tight_layout()
    plt.show()

# Hàm chính để chạy chương trình
def main():
    print("Effect of Focal Length and Pinhole Size")
    visualize_focal_length_and_pinhole_effect()

if __name__ == "__main__":
    main()