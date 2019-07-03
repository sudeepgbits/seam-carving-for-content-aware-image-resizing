# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 22:35:07 2017

@author: SUDEEP
"""

import functions as function
import cv2
import numpy as np
import os
from os import path
from glob import glob

### image reduce inputs###
image_reduction = 350 #number of seams to be reduced from figure 5 350

### image retargate inputs  200  110##
desired_columns = 200
desired_rows = 110


### image retargate inputs  119##
seams_insert = 119


""" function for image reduction"""
img5 = cv2.imread('fig5.png',cv2.IMREAD_COLOR)
print('Processing Image Reduction')
[cropped_image,minima,energy,seam_coordinates] = function.seamremove(img5,image_reduction)
cv2.imwrite('fig5_output.png',cropped_image)



""" function for image enlarge"""
img8 = cv2.imread('fig8.png',cv2.IMREAD_COLOR)
print('Processing Image enlargement')
[image_new2,minima,min_energy,seam_coordinates] = function.seamremove(img8,seams_insert)
[enlarged, image_seam] = function.seam_insert(img8,seam_coordinates,seams_insert)
cv2.imwrite('fig8c_output.png',image_seam)
cv2.imwrite('fig8d.png',enlarged)

img8 = cv2.imread('fig8d.png',cv2.IMREAD_COLOR)
print('Processing Image enlargement')
[image_new2,minima,min_energy,seam_coordinates] = function.seamremove(img8,seams_insert)
[enlarged, image_seam] = function.seam_insert(img8,seam_coordinates,seams_insert)
cv2.imwrite('fig8e.png',enlarged)


""" function for image retarget"""
img2 = cv2.imread('fig7.png',cv2.IMREAD_COLOR)
print('Processing Image Retarget')
[image_new, map] = function.transportmat(img2,desired_rows,desired_columns)
retargated = image_new
cv2.imwrite('fig7_output.png',retargated)
cv2.imwrite('transport map.png',map)

