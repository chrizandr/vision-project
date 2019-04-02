"""Main algorithm."""
from radon import radon_projection, inverse_radon_reconstruction
from initial_estimation import initialize_LK
from k_theta import compute_ktheta
from noise_free_l import compute_l_zero
from directional_filter import directional_filter, apply_filter
import pickle
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
import math
import numpy as np
import pdb


def k_estimation(b0, Nf=10):
    """Estimate blur kernel."""
    b1 = rescale(b0, 1.0/2, multichannel=False, anti_aliasing=False)
    print("Finding initial estimate, l1, k1 --> l0")
    l1, k1, _ = initialize_LK(b1)
    l0 = resize(l1, b0.shape, preserve_range=True, anti_aliasing=False)
    k0 = k1
    theta_arr = []
    for i in range(0, Nf):
        angle = np.rad2deg(i*np.pi/Nf)+math.degrees(math.pi/2)
        theta_arr.append(angle)
    theta_arr = np.array(theta_arr)

    prev_k0 = np.zeros(k0.shape)

    count = 0
    for m in range(10):
        verbose = False
        print("Iteration number: ", count)
        b_theta_arr = []
        for i in range(0, Nf):
            print("Filtering b0 --> b_theta; filter: ", i)
            dfilter = directional_filter((i*180)/Nf, 31)
            b_theta = apply_filter(b0, dfilter)
            b_theta_arr.append(b_theta)
        b_theta_arr = np.array(b_theta_arr)

        r_transforms = []
        for i in range(0, Nf):
            print("Computing b0 --> k_theta; filter: ", i)
            k_theta = compute_ktheta(b_theta_arr[i], l0, k0, verbose=verbose)
            r_transform = radon_projection(k_theta, [theta_arr[i]])
            r_transforms.append(r_transform[:, 0])

        r_transforms = np.array(r_transforms)
        print("Radon reconstruction [k_theta] --> k0")
        k0 = inverse_radon_reconstruction(r_transforms.T, theta_arr)

        print("Compute final latent image [b0, k0] --> l0")
        l0 = compute_l_zero(b0, l0, k0)

        error = np.linalg.norm(prev_k0-k0)
        if error < 0.05:
            break

        print("Change from previous k0[i-1] - k[i]: ", error)
        count += 1

    return k0, l0, error


if __name__ == "__main__":
    blur_img = imread('test.jpg', as_gray=True)
    k0, l0, error = k_estimation(blur_img)
    pickle.dump((k0, l0, error), open("final.pkl", "wb"))
