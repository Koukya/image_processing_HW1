import cv2
import numpy as np

def color_road_area(image):
    # 將圖片轉換為 HSV 色彩空間以便更容易選取顏色範圍
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv_image', hsv_image)
    # 設定道路顏色範圍（根據圖片調整，這裡假設道路是深灰色）
    lower_gray = np.array([0, 0, 0])   # HSV中的深灰色範圍的低限
    upper_gray = np.array([180, 55, 220])  # HSV中的深灰色範圍的高限
    mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
    
    # 使用形態學閉操作來去除小雜訊並連接道路區域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    road_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 將選中的道路區域塗上顏色（例如藍色）
    result_image = image.copy()
    result_image[road_mask > 0] = [0, 180, 0]  # 使用藍色塗上道路區域

    return result_image

# 讀取圖片
image_original = cv2.imread('test.jpg')

# 塗上道路區域
colored_image = color_road_area(image_original)

# 顯示結果
cv2.imshow('image_original', image_original)
cv2.imshow('Road Area Colored', colored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果需要保存結果
cv2.imwrite('test_road_colored.jpg', colored_image)
