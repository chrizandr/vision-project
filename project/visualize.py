"""Visualisation."""
from radon import radon_projection
from directional_filter import directional_filter, apply_filter
import math
import matplotlib.pyplot as plt
import pickle
from skimage.io import imread


if __name__ == "__main__":
    blur_img = imread('test.jpg', as_gray=True)
    kernel = directional_filter(90, 31)
    b_theta = apply_filter(blur_img, kernel)
    # plt.subplot(121)
    # plt.imshow(blur_img, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(b_theta, cmap='gray')
    # plt.show()
    L, K = pickle.load(open("test_init.pkl", "rb"))
    angle = 90+math.degrees(math.pi/2)
    # k_theta = compute_ktheta(b_theta, L, K, verbose=True)
    # pdb.set_trace()
    k_theta = pickle.load(open("kt_90.pkl", "rb"))
    print(((k_theta-K)**2).sum())
    r_transform1 = radon_projection(k_theta, [angle])
    r_transform1_90 = radon_projection(k_theta, [60])

    r_transform2 = radon_projection(K, [angle])
    r_transform2_90 = radon_projection(K, [60])
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(r_transform1[:, 0])
    ax1.plot(r_transform1_90[:, 0])
    ax2.plot(r_transform2[:, 0])
    ax1.plot(r_transform2_90[:, 0])
    plt.show()
