import cv2 
import numpy

image_Shape = [0,0]
Kernal_size = 3

def image_pre_process(image):   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_Shape = image.shape #得到圖片像素大小
    rows, cols = image_Shape
    return image, image_Shape, rows, cols

def image_array():
    value = [[0]*image_cols for i in range(image_rows)]
    for y in range(image_rows):
        row = y 
        for x in range(image_cols):
            col = x
            value[row][col] = image_gray[row,col]
    return value

def kernals(size,value,image):
    for y in range(image_rows-size+1):
        row = y 
        for x in range(image_cols-size+1):
            col = x 
            temp = 0
            for z in range(size):
                for q in range(size):
                    temp += image[row+z][col+q]
            value[row][col] = temp//9
    return value

def draw(image,value):
    array = numpy.array(image)
    print(array[200,200])
    print(array[200,200][0],array[200,200][1],array[200,200][2])
    for y in range(image_rows):
        row = y
        for x in range(image_cols):
            col = x
            if value[row][col] <=40 and array[y,x][0] <= 50 and array[y,x][1] <= 50 and array[y,x][2] <= 50 :
                image[row,col] = [255, 0, 0]
    return  image

image_original = cv2.imread('test.jpg')
image_afterdraw = image_original
image_gray, image_Shape, image_rows, image_cols = image_pre_process(image_original)
value = image_array()
value = kernals(Kernal_size,value,image_gray)
print(image_original[400,350]+1)
image_afterdraw = draw(image_afterdraw,value)

#print (image_Shape)
#print (value)
#print(value[400][350],image_original[400,350])
#print(image_original[400,350]+1)
#print(image_afterdraw[400,350]+1)
#cv2.imshow('a', image_afterdraw)
#cv2.imshow('result', image_gray)
#cv2.waitKey(0)



