#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 27 00:52:45 2019

"""

import numpy as np
import pdb
import matplotlib.pyplot as plt


from skimage.io import imread,imsave
from skimage import data_dir
from skimage.transform import radon, rescale
from skimage.transform import iradon
import math

#RADON

#PROVIDE K-theta in place of image
def radon_projection(image , theta_arr):


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

    ax1.set_title("Original")
    ax1.imshow(image, cmap=plt.cm.Greys_r)

    #theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    sinogram = radon(image, theta_arr ,  circle=True)
    print(sinogram)

    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")

    ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
               extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

    #ax2.imshow(sinogram, cmap=plt.cm.Greys_r,extent=(0, 180,-sinogram.shape[0]/2.0, sinogram.shape[0]/2.0) , aspect ='auto')

    fig.tight_layout()
    plt.show()
    return sinogram

#INVERSE RADON

def inverse_radon_reconstruction(sinogram, theta_arr):

    reconstruction_fbp = iradon(sinogram, theta_arr, circle=True)
    plt.imshow(reconstruction_fbp, 'gray')
    plt.show()
    error = reconstruction_fbp - image
    print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error**2)))

    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True, sharey=True)
    ax1.set_title("Reconstruction\nFiltered back projection")
    ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    ax2.set_title("Reconstruction error\nFiltered back projection")
    ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
    plt.show()

# image = imread(data_dir + "/phantom.png", as_gray=True)
# image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
d = np.sqrt(x*x+y*y)
sigma, mu = 1.0, 0.0
image = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
num_angles = 36
theta_arr = np.zeros(num_angles)
for i in range(num_angles):
    theta_arr[i] = np.rad2deg(i*np.pi/num_angles)+math.degrees(math.pi/2)


sin = radon_projection(image, theta_arr)
# pdb.set_trace()/
reconstruction = inverse_radon_reconstruction(sin,theta_arr)
