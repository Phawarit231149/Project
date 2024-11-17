# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:07:35 2023

@author: TNP
"""

import matplotlib.pyplot as plt
from sys import exit
import pandas as pd
import numpy as np
from skimage.feature import   graycomatrix, graycoprops
from skimage.io import imread
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score

#----------------------------- Histrogram -------------------------------------
dirname = "D:/Project/brain-tumor"
imagedirname = dirname + "/images"
image_number= '212'
my_image=imread(imagedirname + "/Image" + image_number + ".jpg" , as_gray=True )
my_image = (my_image*255).astype(int)
plt.figure()
plt.imshow(my_image, cmap="gray")

# create a histogram          

h = np.zeros( 256 )             
for x in range(my_image.shape[0]):      
    for y in range(my_image.shape[1]):            
        i = my_image[x,y]                  
        h[i] = h[i]+1

#take average of 5 neighbors
hl = np.zeros(256)
hl[0:255] = h[1:256]   #shift left
hl[255]   = hl[254]
#
hll = np.zeros(256)
hll[0:255] = hl[1:256]   #shift left again
hll[255]   = hll[254]
#
hlll = np.zeros(256)
hlll[0:255] = hll[1:256]   #shift left again
hlll[255]   = hlll[254]
#
hllll = np.zeros(256)
hllll[0:255] = hlll[1:256]   #shift left again
hllll[255]   = hllll[254]
#
hr = np.zeros(256)
hr[1:256] = h[0:255]   #shift right
hr[0]     = hr[1]

hrr = np.zeros(256)
hrr[1:256] = hr[0:255]   #shift right again
hrr[0]     = hrr[1]

hrrr = np.zeros(256)
hrrr[1:256] = hrr[0:255]   #shift right again
hrrr[0]     = hrrr[1]

hrrrr = np.zeros(256)
hrrrr[1:256] = hrrr[0:255]   #shift right again
hrrrr[0]     = hrrrr[1]
#  take average
h_new = (h + hl + hll + hlll + hr + hrr  + hrrr) / 9

plt.figure()
plt.stairs( h_new[10:255] )        
plt.show()

#------------------------------- bin-------------------------------------------
my_bins=np.histogram(my_image,bins=8)
new_bins=my_bins[0][1:8]   
max_bins=np.argmax( new_bins ) 

plt.figure()
plt.stairs( new_bins )  
plt.show()

#------------------------ tumor(guess) before norm. ---------------------------
tumor_guess=0  
if (max_bins < 5) and ( new_bins[6] >= 80):
    tumor_guess=1

#------------------------ after normalize pics --------------------------------

sum_pixel_values = 0
n_pixels = 0
for x in range(my_image.shape[0]):        
    for y in range(my_image.shape[1]):
        if my_image[x,y] >= 1:
            n_pixels = n_pixels + 1
            sum_pixel_values = sum_pixel_values + my_image[x,y]
avg_pixel_values = sum_pixel_values/n_pixels
avg_pixel_values = np.floor(avg_pixel_values)

new=144
avg=avg_pixel_values 
mod=my_image
m1 = (255-new)/(255-avg)
m2 = (new-32)/(avg-32)
for x in range(my_image.shape[0]):        
    for y in range(my_image.shape[1]):
        if mod[x,y] >= avg:
            mod[x,y] = np.floor(255+(m1*(mod[x,y]-255)))
          
        elif mod[x,y] >= 32:
            mod[x,y] = np.floor(32+(m2*(mod[x,y]-32)))

plt.figure()
plt.stairs( h[10:255])
plt.figure()
plt.imshow(my_image, cmap="gray")

h = np.zeros( 256 )             
for x in range(my_image.shape[0]):        
    for y in range(my_image.shape[1]):            
        i = my_image[x,y]                  
        h[i] = h[i]+1
 
plt.figure()
plt.stairs(h[10:255])
plt.show()

my_bins=np.histogram(my_image,bins=8)
new_bins=my_bins[0][1:8]
max_bins=np.argmax( new_bins )

###############################################################################
h = np.zeros( 256 )             
for x in range(my_image.shape[0]):      
    for y in range(my_image.shape[1]):            
        i = my_image[x,y]                  
        h[i] = h[i]+1

#take average of 5 neighbors
hl = np.zeros(256)
hl[0:255] = h[1:256]   #shift left
hl[255]   = hl[254]
#
hll = np.zeros(256)
hll[0:255] = hl[1:256]   #shift left again
hll[255]   = hll[254]
#
hlll = np.zeros(256)
hlll[0:255] = hll[1:256]   #shift left again
hlll[255]   = hlll[254]
#
hllll = np.zeros(256)
hllll[0:255] = hlll[1:256]   #shift left again
hllll[255]   = hllll[254]
#
hr = np.zeros(256)
hr[1:256] = h[0:255]   #shift right
hr[0]     = hr[1]

hrr = np.zeros(256)
hrr[1:256] = hr[0:255]   #shift right again
hrr[0]     = hrr[1]

hrrr = np.zeros(256)
hrrr[1:256] = hrr[0:255]   #shift right again
hrrr[0]     = hrrr[1]

hrrrr = np.zeros(256)
hrrrr[1:256] = hrrr[0:255]   #shift right again
hrrrr[0]     = hrrrr[1]
#  take average
h_new = (h + hl + hll + hlll + hr + hrr  + hrrr) / 9
plt.figure()
plt.stairs( h_new[10:255] )        
plt.show()
###############################################################################

plt.figure()
plt.stairs( new_bins )  
plt.show()



after_tumor=0
if (max_bins < 5) and ( new_bins[6] >= 80):
    after_tumor=1

print('old result =',tumor_guess,' , new result =',after_tumor)    
###############################################################################

bin_image=np.zeros(my_image.shape).astype(np.uint8) 
for x in range(bin_image.shape[0]):
    for y in range(bin_image.shape[1]) :
     if mod[x,y] >= 224 :
        bin_image[x,y]=255
plt.imshow( bin_image, cmap='gray')

import cv2

#control_gray = cv2.cvtColor(binn_img,)
contours,hierarchy = cv2.findContours(bin_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
big_contour = max(contours,key=cv2.contourArea)
 #fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree
ellipse =  cv2.fitEllipse(big_contour)

BacktoRGB = cv2.cvtColor(bin_image,cv2.COLOR_GRAY2RGB)

plt.figure()
plt.imshow( BacktoRGB ) 
cv2.ellipse(BacktoRGB,ellipse,(255,0,0),3)
(xc,yc),(d1,d2,),angle = ellipse

plt.figure()
plt.imshow( BacktoRGB )

ellipse_aspect_ratio = d2/d1
print('ellipse_aspect_ratio: ',ellipse_aspect_ratio)

binbacktorgb= cv2.cvtColor(bin_image,cv2.COLOR_GRAY2RGB)
cv2.drawContours(binbacktorgb, contours,-1,(255,0,0))
plt.figure()
plt.imshow( binbacktorgb )

kernel = np.ones((3,3),np.uint8)
eroded_img = cv2.erode(bin_image,kernel,iterations = 1)
dilated_img = cv2.dilate(eroded_img,kernel,iterations = 1)

b_pixels =0              
x_center = 0 ; y_center = 0                
for x in range(bin_image.shape[1]):        
    for y in range(bin_image.shape[0]):   
        if bin_image[y,x] != 0:
            x_center = x_center + x
            y_center = y_center + y
            b_pixels=b_pixels+1
x_center = x_center/b_pixels
y_center = y_center/b_pixels
area = (b_pixels/n_pixels)*100
print("center of mass at x", round(x_center,3), ',y : ' ,round(y_center,3) )
print("area =", round(area,3),"% of the brain image")

