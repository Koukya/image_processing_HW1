import cv2 
import numpy
from collections import Counter

def Guass(image,level):
    kernal_size = level
    guass_img = cv2.GaussianBlur(image , (kernal_size , kernal_size), 0)
    return guass_img

def kernal_operation(center, neighbor):
    return 1 if neighbor >= center else 0

def lbp(kernal_size, image, top):
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

def draw(image1,image2,top):
    rows, cols = image1.shape
    for y in range(0, rows):
        for x in range(0, cols):
            if((image1[y,x] == top[0][0] or image1[y,x] == top[1][0] or image1[y,x] == top[2][0]) and image_gray[y,x] <= 100 and check_diff(image_original[y,x] , 10) == True):
                image2[y,x] = [0,0,255]
    return image2

def check_diff(arr,D):
    arr.sort()
    
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] > D:
            return False 
    return True  


image_original = cv2.imread('pic/test.jpg')
image_copy = image_original.copy()
image_gray = cv2.cvtColor(image_original,cv2.COLOR_BGR2GRAY)
image_guass = Guass(image_gray,5)
image_lbp,top= lbp(3,image_guass,5)
image_draw = draw(image_lbp,image_copy,top)


cv2.imshow('Ori',image_original)
cv2.imshow('Gray',image_gray)
cv2.imshow('Guass',image_guass)
cv2.imshow('LBP',image_lbp)
cv2.imshow('Draw',image_draw)
cv2.imwrite('draw/ori.jpg',image_original)
cv2.imwrite('draw/gray.jpg',image_gray)
cv2.imwrite('draw/guass.jpg',image_guass)
cv2.imwrite('draw/lbp.jpg',image_lbp)
cv2.imwrite('draw/draw.jpg',image_draw)


cv2.waitKey(0)
cv2.destroyAllWindows()
