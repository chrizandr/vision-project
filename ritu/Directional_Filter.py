import numpy as np
import cv2

gaussian_filter = np.zeros((30,30),np.float32)

t = 1
theta = np.rad2deg(np.pi/36)
#gauss_weight = 0
sigma = 30

for i in range (0,30):
    for j in range(0,30):
        x,y =  np.array([i,j]) + t*np.array([np.cos(theta),np.sin(theta)])
        #print(np.array([i,j]), np.array([np.cos(theta),np.sin(theta)]))
        
        #print(x,y)
        if( 0<=x<=30 and 0<= y<=30):
            print('t',t, '\t')
            print('x,y',x,y)
            gaussian_filter[i,j] = np.exp(-np.square(t)/(2*np.square(sigma)))
        t = t+1
        
print(gaussian_filter)        
        
        
