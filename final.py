import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_road_mask(hsv_image):
    # 設定灰色到黑色的HSV範圍
    lower_bound = np.array([0, 0, 50])       # Hue可以是任意，飽和度低，亮度範圍從50到100
    upper_bound = np.array([180, 50, 100])   # Hue最大為180，飽和度最大50，亮度範圍到100

    # 創建遮罩
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask

def apply_mask(image, mask):
    # 使用遮罩來過濾圖片，顯示馬路區域
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def plot_histogram(hsv_image):
    # 計算並顯示HSV直方圖
    # 計算Hue、Saturation、Value的直方圖
    hist_hue = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])  # Hue直方圖
    hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])  # Saturation直方圖
    hist_value = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])  # Value直方圖

    # 顯示直方圖
    plt.figure(figsize=(10, 7))
    plt.subplot(3, 1, 1)
    plt.plot(hist_hue, color='r')
    plt.title('Hue Histogram')
    plt.xlim(0, 180)

    plt.subplot(3, 1, 2)
    plt.plot(hist_saturation, color='g')
    plt.title('Saturation Histogram')
    plt.xlim(0, 256)

    plt.subplot(3, 1, 3)
    plt.plot(hist_value, color='b')
    plt.title('Value Histogram')
    plt.xlim(0, 256)

    plt.tight_layout()
    plt.show()

# 1. 讀取圖片
image = cv2.imread('pic/test.jpg')  # 更新圖片路徑
original_image = image.copy()

# 2. 轉換為HSV色彩空間
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. 創建顏色遮罩（灰色到黑色範圍）
mask = create_road_mask(hsv_image)

# 4. 找出遮罩的輪廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. 在原圖上標記矩形輪廓（使用紅色）
for contour in contours:
    if cv2.contourArea(contour) > 500:  # 篩選掉小區域（如果需要的話）
        # 取得每個輪廓的最小外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        # 在原圖上繪製矩形（紅色）
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 6. 顯示最終結果
cv2.imshow('Original Image with Rectangles', original_image)

# 顯示HSV直方圖
plot_histogram(hsv_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
