import matplotlib.pyplot as plt
from sys import exit
import pandas as pd
import numpy as np
from skimage.feature import   graycomatrix, graycoprops
from skimage.io import imread
from sklearn.metrics import confusion_matrix, accuracy_score

dirname = "D:/Project/brain-tumor"
imagedirname = dirname + "/images"
image_number= "37"
my_image=imread(imagedirname + "/Image" + image_number + ".jpg" , as_gray=True )
my_image = (my_image*255).astype(int) 
plt.figure()
plt.imshow(my_image , cmap="gray")

h=np.zeros(256)
for x in range (my_image.shape[0]):       
    for y in range (my_image.shape[1]):   
        i = my_image[x,y]
        h[i]=h[i]+1

plt.figure()
plt.stairs( h[10:255] ) 
plt.show()

#Shift Left x1
hl = np.zeros(256)
hl[0:255]=h[1:256]
hl[255]=hl[254]

#Shift Left x2
hll = np.zeros(256)
hll[0:255]=hl[1:256]
hll[255]=hll[254]

#Shift Right x1
hr = np.zeros(256)
hr[1:256]=h[0:255]
hr[0]=hr[1]

#Shift RIght x2
hrr = np.zeros(256)
hrr[1:256]=hr[0:255]
hrr[0]=hrr[1]

# take average average 5 neighbours
h_new = (hll + hl + h + hr + hrr) /5

plt.figure()
plt.stairs( h_new[10:255] ) 
plt.show()
