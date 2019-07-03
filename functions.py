# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 22:25:32 2017

@author: SUDEEP
"""

import numpy as np
import scipy as sp
import scipy.signal  # one option for a 2D convolution library
import cv2


def derivative_x(image):
    [x,y,z] = image.shape
    image = np.float64(image)
    deriv_x = np.zeros(shape=(x,y,z))
    for i in range(0,x):
        for j in range(0,y):
            if (i>0) and (i<x-1):
                """
               deriv_x[i,j,0] = (np.float64(image[i+1,j,0] - image[i,j,0])) * (np.float64(image[i+1,j,0] - image[i,j,0]))
               deriv_x[i,j,1] = (np.float64(image[i+1,j,1] - image[i,j,1])) * (np.float64(image[i+1,j,1] - image[i,j,1]))
               deriv_x[i,j,2] = (np.float64(image[i+1,j,2] - image[i,j,2])) * (np.float64(image[i+1,j,2] - image[i,j,2]))
               """
                deriv_x[i,j,0] = abs((image[i+1,j,0] - image[i-1,j,0]))
                deriv_x[i,j,1] = abs((image[i+1,j,1] - image[i-1,j,1]))
                deriv_x[i,j,2] = abs((image[i+1,j,2] - image[i-1,j,2]))
               
            elif (i==0):
                deriv_x[i,j,0] = 2* abs((image[i,j,0] - image[i-1,j,0])) 
                deriv_x[i,j,1] = 2* abs((image[i,j,1] - image[i-1,j,1])) 
                deriv_x[i,j,2] = 2* abs((image[i,j,2] - image[i-1,j,2])) 
            elif (i == x-1):
                deriv_x[i,j,0] = 2* abs((image[i-1,j,0] - image[i,j,0])) 
                deriv_x[i,j,1] = 2* abs((image[i-1,j,1] - image[i,j,1])) 
                deriv_x[i,j,2] = 2* abs((image[i-1,j,2] - image[i,j,2])) 
           
    return deriv_x
        
def derivative_y(image):
    [x,y,z] = image.shape
    image = np.float64(image)
    deriv_y = np.zeros(shape=(x,y,z))
    for j in range(0,y):
        for i in range(0,x):
            """
           deriv_y[i,j,0] = (np.float64(image[i,j+1,0] - image[i,j,0])) * (np.float64(image[i,j+1,0] - image[i,j,0]))
           deriv_y[i,j,1] = (np.float64(image[i,j+1,1] - image[i,j,1])) * (np.float64(image[i,j+1,1] - image[i,j,1]))
           deriv_y[i,j,2] = (np.float64(image[i,j+1,2] - image[i,j,2])) * (np.float64(image[i,j+1,2] - image[i,j,2]))
           """
            if (j>0) and (j<y-1):
                deriv_y[i,j,0] = abs((image[i,j+1,0] - image[i,j-1,0])) 
                deriv_y[i,j,1] = abs((image[i,j+1,1] - image[i,j-1,1]))
                deriv_y[i,j,2] = abs((image[i,j+1,2] - image[i,j-1,2]))
               
            elif (j==0):
                deriv_y[i,j,0] = 2* abs((image[i,j,0] - image[i,j-1,0])) 
                deriv_y[i,j,1] = 2* abs((image[i,j,1] - image[i,j-1,1])) 
                deriv_y[i,j,2] = 2* abs((image[i,j,2] - image[i,j-1,2])) 
            elif (j==y-1):
                deriv_y[i,j,0] = 2* abs((image[i,j-1,0] - image[i,j,0])) 
                deriv_y[i,j,1] = 2* abs((image[i,j-1,1] - image[i,j,1])) 
                deriv_y[i,j,2] = 2* abs((image[i,j-1,2] - image[i,j,2])) 
    
    
    return deriv_y

def getGradientMagnitude(im):
    #ddepth = cv2.CV_64F
    #pad = cv2.BORDER_REFLECT101
    dx = derivative_x(im)
    dy = derivative_y(im)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 1, dyabs, 1, 0)
    return mag

def getGradientMagnitude2(im):
    ddepth = cv2.CV_64F
    pad = cv2.BORDER_REFLECT101
    dx = cv2.Sobel(im, ddepth, 1, 0, 1, borderType=pad )
    dy = cv2.Sobel(im, ddepth, 0, 1, 1, borderType=pad)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    [m,n,p] = im.shape
    """
    for i in range (0,m):
        for j in range (0,n):
            if i==0:
                mag[i][j][0] = 2* mag[i][j][0]
                mag[i][j[[1] = 2* mag[i][j][1]
                mag[i][j][2] = 2* mag[i][j][2]
            if j ==0:
                mag[i][j][0] = 2* mag[i][j][0]
                mag[i][j][1] = 2* mag[i][j][1]
                mag[i][j][2] = 2* mag[i][j][2]
    """
    return mag

def cumGradMin(grad):
    [m,n,p] = grad.shape
    M = np.zeros(shape=(m,n))
    
    grad_allcolors = np.zeros(shape=(m,n))
    for i in range(0,m):
        for j in range(0,n):
            grad_allcolors[i,j] = (grad[i,j,0]) / 3 + (grad[i,j,1]) / 3 + (grad[i,j,2]) / 3
    M[0,:] = grad_allcolors[0,:] 
    for i in range(1,m):
        for j in range(0,n):
            if (j == 0):
                k = j
            else:
                k = j-1
            if (j<n-1):
                M[i,j] = grad_allcolors[i,j] + min(M[i-1,k], M[i-1,j], M[i-1,j+1])
            else:
                M[i,j] = grad_allcolors[i,j] + min(M[i-1,k], M[i-1,j])
            
    minima = M[m-1,:].argmin()
    min_energy = M[m-1,minima]
    #return [m-1,minima]
    return [M,minima,grad_allcolors,min_energy]

def seamremove(image,number):
    #cv2.imshow('image_before seam',image)
    #cv2.waitKey(0)
    [m,n,p] = image.shape
    coordinate_j = 0
    seam_coordinates = np.zeros(shape=(m,number))
    for o in range(0,number):
        print("Removing seam:" + str(o))  
        gradin = getGradientMagnitude(image)
        [M,minima,grad,min_energy] = cumGradMin(gradin)
        [m,n] = grad.shape
        i = int(m)
        j = minima
        
        #seam_coordinates = np.zeros(shape=(m,2))
        image_new2 = np.zeros(shape=(m,n-1,3))
        image_new2 = np.uint8(image_new2)    
        while(i>0):
            i = i -1
            image_new2[i,:,0] = np.delete(image[i,:,0],j)
            image_new2[i,:,1] = np.delete(image[i,:,1],j)
            image_new2[i,:,2] = np.delete(image[i,:,2],j)
            """
            for l in range(0,n):
                if (l != j):
                    #print('inloop')
                    image_new2[i,t,0] = np.uint8(image[i,l,0])
                    image_new2[i,t,1] = np.uint8(image[i,l,1])
                    image_new2[i,t,2] = np.uint8(image[i,l,2])
                    t = t+1
            """
            #seam_coordinates[i,:] = [i,j]
            seam_coordinates[i,coordinate_j] = j
            #print(i)
            if(j>0):
                minima_temp = M[i,j-1:j+2].argmin()
            else:
                minima_temp = M[i,j:j+2].argmin()
            if(j>0):
                j = j + (minima_temp - 1)
            else:
                j = 0
            #print(j)
            #image[i,j,0] = 0
            #image[i,j,1] = 0
            #image[i,j,2] = 0
        image = image_new2
        coordinate_j = coordinate_j+1   
        #cv2.imwrite("seam removed " + str(o) + ".png" ,image_new2)
        #cv2.waitKey(0)
    return [image_new2,minima,min_energy,seam_coordinates]

"""
def seamremove2(image,number):
    #[m,n] = image.shape
    image_new2 = image
    for t in range(0,number):
        image_new = image_new2
        #cv2.imshow('image_new',image_new)
        #cv2.waitKey(0)
        #grad = getGradientMagnitude(image_new)
        [image_new, coordinates] = backtrace(image_new)
        #cv2.imshow('image_seam2',image_new)
        #cv2.waitKey(0)
        [m,n,p] = image_new.shape
        #print(image_new.shape)
        image_new2 = np.zeros(shape=(m,n-1,p))
        image_new2 = np.uint8(image_new2)
        #image_new2 = image_new[:,1:n-5,:]
        #cv2.imshow('image_new2',image_new)
        #cv2.waitKey(0)
        for i in range(0,m):
            k = 0
            for j in range(0,n):
                if (j != int(coordinates[i,1])):
                    #print('in loop')
                    image_new2[i,k,0] = np.uint8(image_new[i,j,0])
                    image_new2[i,k,1] = np.uint8(image_new[i,j,1])
                    image_new2[i,k,2] = np.uint8(image_new[i,j,2])
                    k = k+1
                #else:
                    #print(int(coordinates[i,1]))
                    #exit(0)
        #print(image_new)
    return image_new2
"""

def transportmat(img2,n1,m1):
    [n,m,p] = img2.shape
    print("IMAGE IS OF SIZE:" +str(n)+","+str(m))
    r = n - n1 + 1
    c = m - m1 + 1
    T = np.zeros(shape=(r,c))
    map = np.zeros(shape=(r,c))
    Image_list = []
    horizontal_seam = 0
    vertical_seam = 0
    for i in range(0,r):
        Image_list.append(list())
    
    image_step = img2
    for i in range(0,r):
        for j in range(0,c):
            print("PROCESSING:" +str(i)+ ","+str(j))
            if ((i==0) and (j==0)):
                T[i][j] = 0
                
                Image_list[i].append(np.uint8(image_step))
            elif ((i == 0) and (j > 0)):
                image_step = Image_list[i][j-1]
                [image_step,minima,Esy,seam_coordinates] = seamremove(image_step,1)
                Image_list[i].append(np.uint8(image_step))
                T[i][j] = Esy
            elif ((i > 0) and (j == 0)):
                #image_step = Image_list[i-1][j]
                [u,v,w] = Image_list[i-1][j].shape
                image_step_tempx = np.zeros(shape=(v,u,w))
                image_step_tempx[:,:,0] = Image_list[i-1][j][:,:,0].transpose()
                image_step_tempx[:,:,1] = Image_list[i-1][j][:,:,1].transpose()
                image_step_tempx[:,:,2] = Image_list[i-1][j][:,:,2].transpose()
                [image_step_tempx2,minima,Esx,seam_coordinates] = seamremove(image_step_tempx,1)
                [u,v,w] = image_step_tempx2.shape
                image_step_x = np.zeros(shape=(v,u,w))
                
                image_step_x[:,:,0] = image_step_tempx2[:,:,0].transpose()
                image_step_x[:,:,1] = image_step_tempx2[:,:,1].transpose()
                image_step_x[:,:,2] = image_step_tempx2[:,:,2].transpose()
                Image_list[i].append(np.uint8(image_step_x))
                T[i][j] = Esx
                #map[i][j] = 1
            else:
                [u,v,w] = Image_list[i-1][j].shape
                image_step_tempx = np.zeros(shape=(v,u,w))
                image_step_y = Image_list[i][j-1]
                #image_step_tempx = Image_list[i-1][j]
                image_step_tempx[:,:,0] = Image_list[i-1][j][:,:,0].transpose()
                image_step_tempx[:,:,1] = Image_list[i-1][j][:,:,1].transpose()
                image_step_tempx[:,:,2] = Image_list[i-1][j][:,:,2].transpose()
                #image_step_x = np.zeros(shape=(u-1,v,w))
                [image_step_y,minima,Esy,seam_coordinates] = seamremove(image_step_y,1)
                [image_step_tempx2,minima,Esx,seam_coordinates] = seamremove(image_step_tempx,1)

                
                Tx = T[i-1][j] + Esx
                Ty = T[i][j-1] + Esy
                if (Tx < Ty): 
                    T[i,j] = Tx      
                    [u,v,w] = image_step_tempx2.shape
                    image_step_x = np.zeros(shape=(v,u,w))
                    image_step_x[:,:,0] = image_step_tempx2[:,:,0].transpose()
                    image_step_x[:,:,1] = image_step_tempx2[:,:,1].transpose()
                    image_step_x[:,:,2] = image_step_tempx2[:,:,2].transpose()
                    Image_list[i].append(np.uint8(image_step_x))
                    #T[i][j] = 1
                    map[i][j] = 1
                else:
                    T[i][j] = Ty
                    Image_list[i].append(np.uint8(image_step_y))
                    
    
    map = transport_map(T)
    return [Image_list[r-1][c-1], map]


def transport_map(T):
    [m,n] = T.shape
    map = np.ones(shape=(m,n))
    col_red = 0
    row_red = 0
    i = m-1
    j = n-1
    while ((i != 0) or (j !=0)):
        
            if (i == m-1) and (j == n-1):
                map[i,j] = 0
            if (i>0) and (j>0):
                if (T[i-1,j] < T[i,j-1]):
                    map[i-1,j] = 0
                    i = i-1
                elif (T[i-1,j] > T[i,j-1]):
                    map[i,j-1] = 0
                    j = j-1
            elif (i == 0):
                map[i,j-1] = 0
                j = j-1
            elif (j ==0):
                map[i-1,j] = 0
                i = i-1
            #print("i =" +str(i))
            #print("j =" +str(j))
    return map
            
    

    
    
    
    
    
def seam_calculations(image,number):
  
    for o in range(0,1):
        
        gradin = getGradientMagnitude(image)
        [M,minima,grad,min_energy] = cumGradMin(gradin)
        [m,n] = grad.shape
        i = int(m)
        j = minima
        myList = []
        myList = M[m-1,:]
        
        MyList_order = [b[0] for b in sorted(enumerate(myList), key=lambda x:x[1])]
        minima = MyList_order[0]
        coordinate_j = 0
        seam_coordinates = np.zeros(shape=(m,number))
        for k in range(0,number):
            i = int(m)
            #print("Inserting seam:" + str(k))
            minima = MyList_order[k]
            j = minima
            
            
            #image_new2 = np.zeros(shape=(m,n+1,3))
            #image_new2 = np.uint8(image_new2)    
            
            while(i>0):
                #print("step" + str(i))
                
                i = i - 1
                #image_new2[i,:,0] = np.insert(image[i,:,0],[j],image[i,j,0])
                #image_new2[i,:,1] = np.insert(image[i,:,1],[j],image[i,j,1])
                #image_new2[i,:,2] = np.insert(image[i,:,2],[j],image[i,j,2])
                """
                for l in range(0,n):
                    if (l != j):
                        #print('inloop')
                        image_new2[i,t,0] = np.uint8(image[i,l,0])
                        image_new2[i,t,1] = np.uint8(image[i,l,1])
                        image_new2[i,t,2] = np.uint8(image[i,l,2])
                        t = t+1
                """
                seam_coordinates[i,coordinate_j] = j
                #print(i)
                if(j>0): 
                    #print("n = " +str(n) + ", j =" +str(j))
                    if (j+1<n):
                        
                        #print(M[i,j-1:j+2])
                        minima_temp = M[i,j-1:j+2].argmin()
                    else:
                        minima_temp = M[i,j-1:j+1].argmin()
                else:
                    minima_temp = M[i,j:j+2].argmin()
                
                if(j>0):
                    j = j + (minima_temp - 1)
                else:
                    j = 0
                #print(j)
                #image[i,j,0] = 0
                #image[i,j,1] = 0
                #image[i,j,2] = 0
            #image = image_new2
            coordinate_j = coordinate_j + 1
            #cv2.imshow('image_seam',image_new2)
            #cv2.waitKey(0)
                
    #cv2.imshow('image_seam',image_new2)
    #cv2.waitKey(0)
    return seam_coordinates

def seam_insert(image,seam_coordinates,number):
    [m,n,p] = image.shape
    
    #image_new1 = np.uint8(image_new2)    

    """
    for t in range(0,number):
        print("Inserting seam:" + str(t))  
        #column = seam_coordinates[:,t]
        if (t>0):
            if (seam_coordinates[0,t] >= seam_coordinates[0,t-1]):
                offset = offset+1
        col_size = col_size+1  
        for i in range(0,m):
            
            column = seam_coordinates[i,t]
            #[l,o,t] = image_new1.shape
            #col_size = o + 1
            
            print(col_size)
            #print("Inserting in column =" + str(column) + "vlue =" + str(image[i,int(column),0]))
            image_new2[i,0:col_size,0] = np.insert(image_new1[i,:,0],[int(column+offset)],image_new1[i,int(column+offset),0] )
            image_new2[i,0:col_size,1] = np.insert(image_new1[i,:,1],[int(column+offset)],image_new1[i,int(column+offset),1] )
            image_new2[i,0:col_size,2] = np.insert(image_new1[i,:,2],[int(column+offset)],image_new1[i,int(column+offset),2] )
            #print(image.shape)
            image_new1 = image_new2
        """
        
    image_new2 = np.uint8(np.zeros(shape=(m,n+number,3)))
    image_seam = np.uint8(np.zeros(shape=(m,n+number,3)))
    
    def sublistHasLess(sublist,elem):
        for e in range(0,len(sublist)):
            if(sublist[e] <= elem):
                sublist = np.delete(sublist,e)
                return [sublist,True]
        return [sublist,False]

    def correct_cords(lis):
        for pos in range(0,len(lis)):
            subl = lis[:pos]
            has = True
            while(has):
                [subl,has] = sublistHasLess(subl,lis[pos])
                if(has):
                    lis[pos] = lis[pos] + 1
                
    for i in range(0,m):
         print("Inserting seam:" + str(i))  
         t = 0
         se = seam_coordinates[i,:] 
         correct_cords(se)
         seam_coordinates_temp = np.sort(se)
         jnew = 0
         for j in range(0,n):
             if(t < len(seam_coordinates_temp)):
                 position_original = int(seam_coordinates_temp[t]) 
            # print("j:"+str(j)+", jnew:"+str(jnew) + " for position:"+ str(position_original))
             if ((position_original>0) and (position_original<n-1)):
                 pre_elem = np.uint8(image[i][position_original-1] / 2 )
                 next_elem = np.uint8(image[i][position_original+1] / 2)
                 element_r = pre_elem[0] + next_elem[0]
                 element_y = pre_elem[1] + next_elem[1]
                 element_b = pre_elem[2] + next_elem[2]
             elif (position_original == 0):
                 pre_elem = np.uint8(image[i][position_original] / 2 )
                 next_elem = np.uint8(image[i][position_original+1] / 2)
                 element_r = pre_elem[0] + next_elem[0]
                 element_y = pre_elem[1] + next_elem[1]
                 element_b = pre_elem[2] + next_elem[2]
             elif (position_original == n-1 ):
                 pre_elem = np.uint8(image[i][position_original] / 2 )
                 next_elem = np.uint8(image[i][position_original-1] / 2)
                 element_r = pre_elem[0] + next_elem[0]
                 element_y = pre_elem[1] + next_elem[1]
                 element_b = pre_elem[2] + next_elem[2]
                 
             if( j == position_original):
                 #print("copying element r=" + str(element_r) + " element y=" + str(element_y) + " element b=" + str(element_b) )
                 image_new2[i][jnew][0] = element_r
                 image_new2[i][jnew][1] = element_y
                 image_new2[i][jnew][2] = element_b
                 
                 image_seam[i][jnew][0] = 0
                 image_seam[i][jnew][1] = 0
                 image_seam[i][jnew][2] = 255
                 
                 jnew = jnew + 1
                 t = t + 1
                 
             if (jnew < n + number ):
                 #if(lm[0] == 0 and lm[1] == 0 and lm[2] == 0):
                     #print("copying element r=" + str(image[i][j][0]) + " element y=" + str(image[i][j][1]) + " element b=" + str(image[i][j][2]) )
                 image_new2[i][jnew][0] = np.uint8(image[i][j][0])
                 image_new2[i][jnew][1] = np.uint8(image[i][j][1])
                 image_new2[i][jnew][2] = np.uint8(image[i][j][2])
                 
                 image_seam[i][jnew][0] = np.uint8(image[i][j][0])
                 image_seam[i][jnew][1] = np.uint8(image[i][j][1])
                 image_seam[i][jnew][2] = np.uint8(image[i][j][2])
                 jnew = jnew + 1
                 
    return [image_new2, image_seam]
    
