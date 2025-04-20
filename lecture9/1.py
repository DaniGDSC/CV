import numpy as np

# Hàm thực hiện phép chiếu pinhole
def pinhole_projection(X, Y, Z, f):
    """
    Chiếu một điểm 3D (X, Y, Z) thành tọa độ 2D (x, y) bằng phép chiếu pinhole.
    
    Parameters:
    - X, Y, Z: Tọa độ 3D của điểm
    - f: Tiêu cự (focal length)
    
    Returns:
    - (x, y): Tọa độ 2D trên mặt phẳng ảnh
    """
    if Z == 0:
        raise ValueError("Z không được bằng 0 trong phép chiếu pinhole!")
    
    x = f * X / Z
    y = f * Y / Z
    return x, y

# Hàm chính để chạy chương trình
def main():
    # Điểm 3D ví dụ
    X, Y, Z = 1, 1, 2
    f = 1  # Tiêu cự
    
    try:
        # Thực hiện phép chiếu pinhole
        x, y = pinhole_projection(X, Y, Z, f)
        
        # In kết quả
        print("Bài tập 1: Simple Pinhole Projection")
        print(f"Điểm 3D: ({X}, {Y}, {Z})")
        print(f"Tiêu cự f: {f}")
        print(f"Tọa độ 2D: ({x}, {y})")
    
    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")

if __name__ == "__main__":
    main()