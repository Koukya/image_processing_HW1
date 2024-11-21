import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import cdist

# 設定參數
lbp_radius = 1  # LBP 半徑
lbp_n_points = 8 * lbp_radius  # LBP 點數
stride = 1  # 搜索步長
kernel_size = 3  # Sobel 核心大小

# 讀取影像
image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# 1. 計算 LBP
lbp_image = local_binary_pattern(image, lbp_n_points, lbp_radius, method="uniform")

# 2. 計算 LBP 直方圖
hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, lbp_n_points + 3), range=(0, lbp_n_points + 2))

# 3. 設定標籤和搜尋區域（這裡可以根據需求調整）
label = np.zeros_like(image)  # 初始化標籤矩陣
search_area = (slice(0, image.shape[0], stride), slice(0, image.shape[1], stride))

# 4. 使用 Sobel 邊緣檢測
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

# 5. 距離計算（根據需求，可用於篩選區域）
distance_threshold = 10  # 假設的距離閾值
distance = cdist([hist], [hist], metric="euclidean")[0][0]  # 自行定義距離比較方式

# 6. 應用距離閾值，標籤化符合條件的區域
if distance < distance_threshold:
    label[search_area] = 1  # 設置標籤

# 顯示結果
cv2.imshow('Original Image', image)
cv2.imshow('LBP Image', lbp_image.astype(np.uint8))
cv2.imshow('Sobel Magnitude', sobel_magnitude.astype(np.uint8))
cv2.imshow('Labeled Area', label.astype(np.uint8) * 255)

cv2.waitKey(0)
cv2.destroyAllWindows()
