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

def lbp(kernal_size,image):
    rows , cols = image.shape
    value = image
    for y in range(kernal_size//2,rows-kernal_size//2-1):
        row = y 
        for x in range(kernal_size//2,cols-kernal_size//2-1):
            col = x 
            value[row,col] = 2**7*kernal_operation(image[row-1,col-1],image[row,col])
            +2**6*kernal_operation(image[row-1,col],image[row,col])
            +2**5*kernal_operation(image[row-1,col+1],image[row,col])
            +2**4*kernal_operation(image[row,col+1],image[row,col])
            +2**3*kernal_operation(image[row+1,col+1],image[row,col])
            +2**2*kernal_operation(image[row+1,col],image[row,col])
            +2**1*kernal_operation(image[row+1,col-1],image[row,col])
            +2**0*kernal_operation(image[row,col-1],image[row,col])
    flat_value = value.flatten()
    counter = Counter(flat_value)
    top3 = counter.most_common(3)
    print(top3)
    return value

def kernal_operation(A,B):
    if(B >= A):
        ans = 1
    else:
        ans = 0
    return ans

image_original = cv2.imread('pic/test.jpg')
image_gray = cv2.cvtColor(image_original,cv2.COLOR_BGR2GRAY)
image_guass = Guass(image_gray,1)
image_lbp = lbp(3,image_guass)
#image_canny = Canny(image_guass)



#cv2.namedWindow('try',0)
#cv2.resizeWindow('try',500,800)
#cv2.createTrackbar('min','try',50,1000,lambda x: x)
#v2.createTrackbar('max','try',100,1000,lambda x: x)



cv2.imshow('Ori',image_original)
cv2.imshow('Gray',image_gray)
cv2.imshow('Guass',image_guass)
cv2.imshow('LBP',image_lbp)
#cv2.imshow('5',road_colored)

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