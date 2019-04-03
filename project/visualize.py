"""Visualisation."""
from radon import radon_projection
from directional_filter import directional_filter, apply_filter
import math
import matplotlib.pyplot as plt
import pickle
from skimage.io import imread
from k_theta import compute_ktheta
import numpy as np
import pdb
from scipy import signal

def normalize(r_transform1):
	r_transform1 = (r_transform1 - np.min(r_transform1))/(np.max(r_transform1) - np.min(r_transform1))
	return r_transform1

if __name__ == "__main__":
    blur_img = imread('test.jpg', as_gray=True)
    theta = 60
    kernel = directional_filter(theta, 31)
    b_theta = apply_filter(blur_img, kernel)
    
    # kernel1 = directional_filter(145, 31)
    # b_theta1 = apply_filter(blur_img, kernel1)
    # plt.subplot(131)
    # plt.axis('off')
    # plt.imshow(blur_img, cmap='gray')
    # plt.title('Input image')
    # plt.subplot(132)
    # plt.axis('off')
    # plt.imshow(b_theta, cmap='gray')
    # plt.title('Blurred image with theta$^\circ$ directional filter')
    # plt.subplot(133)
    # plt.axis('off')
    # plt.title('Blurred image with 145$^\circ$ directional filter')
    # plt.imshow(b_theta1, cmap='gray')
    # plt.show()
    
    L, K = pickle.load(open("test_init.pkl", "rb"))
    angle = theta+math.degrees(math.pi/2)
    # k_theta = compute_ktheta(b_theta, L, K, verbose=True)
    # pdb.set_trace()
    k_theta = pickle.load(open("kt_90.pkl", "rb"))
    
    r_transform1 = radon_projection(k_theta, [angle])
    r_transform1_90 = radon_projection(k_theta, [theta])
    r_transform2 = radon_projection(K, [angle])
    r_transform2_90 = radon_projection(K, [theta])
    
    # r_transform1 = normalize(r_transform1)
    # r_transform1_90 = normalize(r_transform1_90)
    # r_transform2 = normalize(r_transform2)
    # r_transform2_90 = normalize(r_transform2_90)

    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax2.plot(r_transform1[:, 0], label='R$_{θ+π/2}$(K$_{θ}$)')
    # ax2.plot(r_transform2[:, 0], label='R$_{θ+π/2}$(K$_{θ}$)')
    # ax2.set_title(r'Radon Transform along θ+π/2 direction')
    # ax2.legend()
    # ax1.plot(r_transform1_90[:, 0], label='R$_{θ}$(K$_{θ}$)')
    # ax1.plot(r_transform2_90[:, 0], label='R$_{θ}$(K)')
    # ax1.set_title('Radon Transform along θ direction')
    # ax1.legend()
    # plt.show()


    plt.plot(r_transform1[:, 0], label='R$_{θ+π/2}$(K$_{θ}$)')
    plt.plot(r_transform2[:, 0], label='R$_{θ+π/2}$(K$_{θ}$)')
    plt.title(r'Radon Transform along θ+π/2 direction')
    plt.legend()
    plt.show()