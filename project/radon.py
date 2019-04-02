"""Radon projective transform and reconstruction."""

import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import radon
from skimage.transform import iradon
import math


def radon_projection(image, theta_arr):
    # theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta_arr, circle=True)
    return sinogram


def inverse_radon_reconstruction(sinogram, theta_arr):
    reconstruction_fbp = iradon(sinogram, theta_arr, circle=True)
    return reconstruction_fbp


if __name__ == "__main__":
    # Gaussian 2D
    size = 10
    sigma, mu = 1.0, 0.0
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x*x+y*y)
    image = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    # Num of angles
    num_angles = 36
    theta_arr = np.zeros(num_angles)
    for i in range(num_angles):
        theta_arr[i] = np.rad2deg(i*np.pi/num_angles)+math.degrees(math.pi/2)

    sin = radon_projection(image, theta_arr)
    reconstruction, error = inverse_radon_reconstruction(sin, theta_arr)

    plt.imshow(image, 'gray')
    plt.show()

    plt.imshow(reconstruction, 'gray')
    plt.show()
