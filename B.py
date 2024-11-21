import cv2 
import numpy
from collections import Counter

def Guass(image,level):
    kernal_size = level
    guass_img = cv2.GaussianBlur(image , (kernal_size , kernal_size), 0)
    return guass_img

def Canny(image):
    low_threshold = 15
    high_threshold = 550
    canny_img = cv2.Canny(image , low_threshold , high_threshold)
    return canny_img

def kernal_operation(center, neighbor):
    """
    比較中心像素和鄰近像素的值
    如果鄰近像素值大於或等於中心像素值，返回 1；否則返回 0
    """
    return 1 if neighbor >= center else 0

def lbp(kernal_size, image, top):
    """
    計算影像的 Local Binary Pattern (LBP)
    :param kernal_size: 核大小 (LBP 覆蓋區域的直徑)
    :param image: 灰階影像 (2D NumPy 陣列)
    :return: LBP 特徵圖 (2D NumPy 陣列)
    """
    rows, cols = image.shape
    lbp_image = numpy.zeros_like(image, dtype=numpy.uint8)

    # 遍歷每個像素（忽略邊緣以避免越界）
    for y in range(kernal_size // 2, rows - kernal_size // 2):
        for x in range(kernal_size // 2, cols - kernal_size // 2):
            center = image[y, x]
            binary_value = 0

            # 計算 LBP 值
            binary_value += 2**7 * kernal_operation(center, image[y-1, x-1])  # 左上
            binary_value += 2**6 * kernal_operation(center, image[y-1, x])    # 上
            binary_value += 2**5 * kernal_operation(center, image[y-1, x+1])  # 右上
            binary_value += 2**4 * kernal_operation(center, image[y, x+1])    # 右
            binary_value += 2**3 * kernal_operation(center, image[y+1, x+1])  # 右下
            binary_value += 2**2 * kernal_operation(center, image[y+1, x])    # 下
            binary_value += 2**1 * kernal_operation(center, image[y+1, x-1])  # 左下
            binary_value += 2**0 * kernal_operation(center, image[y, x-1])    # 左

            # 儲存 LBP 值
            lbp_image[y, x] = binary_value
    flat_value = lbp_image.flatten()
    counter = Counter(flat_value)
    top3 = counter.most_common(top)
    print(top3)
    return lbp_image,top3

#def draw(image1,image2,top):
    rows, cols = image1.shape
    for y in range(0, rows):
        for x in range(0, cols):
            if((image1[y,x] == top[0][0] or image1[y,x] == top[1][0] or image1[y,x] == top[2][0]) and image_gray[y,x] <= 100 and image_original[y,x][1] < 50):
                image2[y,x] = [0,0,255]
    return image2

def draw(image1, image2, top, threshold=100):
    """
    改良的 draw 函式，新增閾值參數以提高靈活性
    """
    rows, cols = image1.shape
    for y in range(rows):
        for x in range(cols):
            # 新增邊緣條件，確保高光點在特定區域
            if (image1[y, x] in [t[0] for t in top[:3]] 
                and image_gray[y, x] <= threshold ):
                #and image_original[y, x] < 50):
                image2[y, x] = [0, 255, 0]  # 改為綠色標記
    return image2


image_original = cv2.imread('pic/test.jpg')
image_copy = image_original.copy()
image_gray = cv2.cvtColor(image_original,cv2.COLOR_BGR2GRAY)
image_guass = Guass(image_gray,5)
image_lbp,top= lbp(3,image_guass,5)
image_draw = draw(image_lbp,image_copy,top)
#image_canny = Canny(image_guass)



#cv2.namedWindow('try',0)
#cv2.resizeWindow('try',500,800)
#cv2.createTrackbar('min','try',50,1000,lambda x: x)
#v2.createTrackbar('max','try',100,1000,lambda x: x)



cv2.imshow('Ori',image_original)
cv2.imshow('Gray',image_gray)
cv2.imshow('Guass',image_guass)
cv2.imshow('LBP',image_lbp)
cv2.imshow('Draw',image_draw)

#while True:
#    min = cv2.getTrackbarPos('min', 'try')
#    max = cv2.getTrackbarPos('max','try')
#    image_try = cv2.Canny(image_guass , min, max)
#    cv2.imshow('image_try',image_try)
#    cv2.waitKey(10)
    
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite('1.jpg', image_original)
#cv2.imwrite('2.jpg', image_gray)
#cv2.imwrite('3.jpg', image_guass)
#cv2.imwrite('4.jpg', image_canny)
#cv2.imwrite('5.jpg', image_canny)