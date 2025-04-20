import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# Hàm tính gradient và góc hướng
def compute_gradients(image):
    # Chuyển sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)
    
    # Tính gradient theo x và y
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Tính độ lớn và góc của gradient (sử dụng unsigned gradients 0-180°)
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx) * (180 / np.pi) % 180
    
    return magnitude, angle

# Hàm tính histogram gradient cho một ô (vectorized)
def compute_cell_histogram_vectorized(magnitude, angle, n_bins=9):
    bin_width = 180 / n_bins
    
    # Tính bin index cho mỗi pixel
    bin_indices = np.floor(angle / bin_width).astype(np.int32)
    
    # Tính phần dư để phân phối tuyến tính giữa các bin
    bin_remainder = (angle % bin_width) / bin_width
    
    # Tạo histogram rỗng
    histogram = np.zeros(n_bins)
    
    # Tính toán histograms sử dụng numpy
    for bin_idx in range(n_bins):
        # Mask các pixel thuộc bin này
        mask = (bin_indices == bin_idx)
        # Cộng vào bin hiện tại (weighted by 1-remainder)
        histogram[bin_idx] += np.sum(magnitude[mask] * (1 - bin_remainder[mask]))
        
        # Cộng vào bin tiếp theo (weighted by remainder)
        next_bin = (bin_idx + 1) % n_bins
        mask = (bin_indices == bin_idx)
        histogram[next_bin] += np.sum(magnitude[mask] * bin_remainder[mask])
    
    return histogram

# Hàm tính histogram gradient cho một ô (dùng loop - chậm hơn nhưng dễ hiểu)
def compute_cell_histogram(magnitude, angle, n_bins=9):
    h, w = magnitude.shape
    bin_width = 180 / n_bins
    histogram = np.zeros(n_bins)
    
    for i in range(h):
        for j in range(w):
            mag = magnitude[i, j]
            ang = angle[i, j]
            
            # Xác định bin
            bin_idx = int(ang // bin_width)
            bin_idx_next = (bin_idx + 1) % n_bins
            
            # Phân phối tuyến tính giữa 2 bin
            fraction = (ang % bin_width) / bin_width
            histogram[bin_idx] += mag * (1 - fraction)
            histogram[bin_idx_next] += mag * fraction
    
    return histogram

# Hàm tính HOG descriptor
def compute_hog_descriptor(image, cell_size=8, block_size=2, n_bins=9):
    # Đảm bảo image là ảnh BGR
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Tính gradient
    magnitude, angle = compute_gradients(image)
    h, w = magnitude.shape
    
    # Tính số lượng cells
    cells_y = h // cell_size
    cells_x = w // cell_size
    
    # Cắt ảnh để đảm bảo kích thước là bội số của cell_size
    magnitude = magnitude[:cells_y*cell_size, :cells_x*cell_size]
    angle = angle[:cells_y*cell_size, :cells_x*cell_size]
    
    # Tạo mảng để lưu histogram của mỗi cell
    cell_histograms = np.zeros((cells_y, cells_x, n_bins))
    
    # Tính histogram cho mỗi cell
    for cy in range(cells_y):
        for cx in range(cells_x):
            y_start = cy * cell_size
            x_start = cx * cell_size
            
            # Lấy vùng ô cell_size x cell_size
            cell_mag = magnitude[y_start:y_start+cell_size, x_start:x_start+cell_size]
            cell_ang = angle[y_start:y_start+cell_size, x_start:x_start+cell_size]
            
            # Tính histogram cho ô
            cell_histograms[cy, cx] = compute_cell_histogram(cell_mag, cell_ang, n_bins)
    
    # Tạo khối và chuẩn hóa
    descriptor = []
    # Lưu vị trí các block để visualization
    blocks = []
    
    for by in range(cells_y - block_size + 1):
        for bx in range(cells_x - block_size + 1):
            # Lưu vị trí block
            blocks.append((bx * cell_size, by * cell_size))
            
            # Lấy histogram của khối block_size x block_size
            block_hist = cell_histograms[by:by+block_size, bx:bx+block_size].flatten()
            
            # Chuẩn hóa L2-norm
            norm = np.linalg.norm(block_hist)
            if norm > 0:
                block_hist = block_hist / (norm + 1e-6)  # Thêm epsilon để tránh chia cho 0
                
            descriptor.extend(block_hist)
    
    return np.array(descriptor), blocks, cell_size, block_size

# Hàm vẽ HOG descriptor lên ảnh
def visualize_hog(image, blocks, cell_size, block_size):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Vẽ các block
    for (x, y) in blocks:
        rect = patches.Rectangle((x, y), 
                                 cell_size * block_size, 
                                 cell_size * block_size, 
                                 linewidth=1, 
                                 edgecolor='r', 
                                 facecolor='none')
        ax.add_patch(rect)
    
    ax.set_title('HOG Visualization')
    ax.axis('off')
    return fig

# Kiểm tra kích thước ảnh
def check_image_size(image, cell_size):
    h, w = image.shape[:2]
    if h % cell_size != 0 or w % cell_size != 0:
        print(f"Warning: Image dimensions ({w}x{h}) are not multiples of cell_size ({cell_size}).")
        print(f"The image will be cropped to {(w//cell_size)*cell_size}x{(h//cell_size)*cell_size}")

# Hàm chính
def main():
    # Đọc ảnh
    image_path = 'img/dog.jpg'  # Cập nhật đường dẫn đến ảnh của bạn
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read the image. Check the path: {image_path}")
        
        # Kiểm tra kích thước ảnh
        cell_size = 8
        check_image_size(image, cell_size)
        
        # Tính HOG descriptor
        hog_descriptor, blocks, cell_size, block_size = compute_hog_descriptor(
            image, cell_size=cell_size, block_size=2, n_bins=9)
        
        # Hiển thị ảnh gốc
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Hiển thị HOG visualization
        visualize_hog(image, blocks, cell_size, block_size)
        plt.tight_layout()
        plt.show()
        
        # In descriptor
        print("HOG Descriptor:")
        print(hog_descriptor[:20], "...") # In 20 phần tử đầu tiên
        print("Kích thước descriptor:", hog_descriptor.shape)
        
        # Vẽ biểu đồ HOG descriptor
        plt.figure(figsize=(12, 4))
        plt.bar(range(min(100, len(hog_descriptor))), hog_descriptor[:100])
        plt.title('HOG Descriptor (first 100 values)')
        plt.xlabel('Feature Index')
        plt.ylabel('Normalized Magnitude')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()