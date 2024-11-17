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
image_number= "37"
my_image=imread(imagedirname + "/Image" + image_number + ".jpg" , as_gray=True )
my_image = (my_image*255).astype(int) #astype(int) = เปลี่ยนข้อมูลเป็น int

my_bins=np.histogram(my_image,bins=8)
new_bins=my_bins[0][1:8]    ### bins is 2d marix so we type [0] to analyze only 1st dimention
max_bins=np.argmax( new_bins )  ### which bins has most pixels

plt.figure()
plt.stairs( new_bins )  
plt.show()

tumor=0
if (max_bins <= 5) and ( new_bins[6] >= 40):
    tumor=1