#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2


#reading the image 

image = cv2.imread('index.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#plotting the image
plt.imshow(image)

#saving image
cv2.imwrite('test_write2.jpg',image)



###########
image = cv2.imread('index.jpg') 
#converting image to Gray scale 
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#plotting the grayscale image
plt.imshow(gray_image) 
cv2.imwrite('test_gray_image.jpg',gray_image)
#converting image to HSV format
hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#plotting the HSV image
plt.imshow(hsv_image)
cv2.imwrite('test_hsv_image.jpg',hsv_image)

#converting image to size (100,100,3) 
smaller_image = cv2.resize(image,(100,100),interpolation=cv2.INTER_LINEAR) 
#plot the resized image
plt.imshow(smaller_image)
cv2.imwrite('test_smaller_image.jpg',smaller_image)


rows,cols = image.shape[:2] 
#(col/2,rows/2) is the center of rotation for the image 
# M is the cordinates of the center 
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1) 
dst = cv2.warpAffine(image,M,(cols,rows)) 
cv2.imwrite('test_dst.jpg',dst)


#shifting the image 100 pixels in both dimensions
M = np.float32([[1,0,-100],[0,1,-100]]) 
dst2 = cv2.warpAffine(image,M,(cols,rows)) 
cv2.imwrite('test_dst2.jpg',dst2)

gray_image = cv2.imread('index.jpg', 0)

ret,thresh_binary = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
ret,thresh_binary_inv = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY_INV)
ret,thresh_trunc = cv2.threshold(gray_image,127,255,cv2.THRESH_TRUNC)
ret,thresh_tozero = cv2.threshold(gray_image,127,255,cv2.THRESH_TOZERO)
ret,thresh_tozero_inv = cv2.threshold(gray_image,127,255,cv2.THRESH_TOZERO_INV)

#DISPLAYING THE DIFFERENT THRESHOLDING STYLES
names = ['Oiriginal Image','BINARY','THRESH_BINARY_INV','THRESH_TRUNC','THRESH_TOZERO','THRESH_TOZERO_INV']
#images = gray_image,thresh_binary,thresh_binary_inv,thresh_trunc,thresh_tozero,thresh_tozero_inv

#for i in range(6):
  
cv2.imwrite('test_thresh_binary2.jpg',thresh_binary)


edges = cv2.Canny(image,100,200) 
cv2.imwrite('test_edges.jpg',edges)
    



