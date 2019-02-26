import numpy as np
import cv2

gaussian_filter = np.zeros((30,30),np.float32)

def directional_filter(theta):

    t = 1
   
    #gauss_weight = 0
    sigma = 30
    
    for i in range (0,30):
        for j in range(0,30):
            x,y =  np.array([i,j]) + t*np.array([np.cos(theta),np.sin(theta)])
            #print(np.array([i,j]), np.array([np.cos(theta),np.sin(theta)]))
            
            #print(x,y)
            if( 0<=x<=30 and 0<= y<=30):
                #print('t',t, '\t')
                #print('x,y',x,y)
                gaussian_filter[i,j] = np.exp(-np.square(t)/(2*np.square(sigma)))
            t = t+1
            
    #print(gaussian_filter)        
        
 
series_gaussian = np.zeros((36,30,30),np.float32)    
print(series_gaussian.shape)
for i in range(0,36):       
    theta = np.rad2deg(i*np.pi/36)
    series_gaussian[i]=directional_filter(theta)
    
print(series_gaussian[35].shape)   