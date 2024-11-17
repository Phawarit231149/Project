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

dirname = "D:/Project/brain-tumor"
imagedirname = dirname + "/images"
image_number= "324" ###You can specify the images number in this part
my_image=imread(imagedirname + "/Image" + image_number + ".jpg" , as_gray=True )
my_image = (my_image*255).astype(int) 

my_bins=np.histogram(my_image,bins=8)
new_bins=my_bins[0][1:8]    
max_bins=np.argmax( new_bins )

plt.figure()
plt.stairs( new_bins )  
plt.show()

tumor=0
if (max_bins <= 5) and ( new_bins[6] >= 40):
    tumor=1

if ( tumor == 0 ):
    print ("There is no tumor in this picture")
else :
    print ("There is tumor in this picture")
