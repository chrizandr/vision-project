from randon import *
from initial_estimation import *
from k_theta import *
from noise_free_l import *
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
import numpy as np
import pdb

def k_estimation(b0, Nf=36):
	b1 = rescale(b0, 1.0/2, multichannel=False, anti_aliasing=False)
	l1, k1, _ =initialize_LK(b1)
	l0 = rescale(l10, 2.0, multichannel=False, anti_aliasing=False)
	k0 = k1
	theta_arr = []
	for i in range(0, Nf):
		angle = np.rad2deg(i*np.pi/Nf)+math.degrees(math.pi/2)
		theta_arr.append(angle)
	theta_arr = np.array(theta_arr)
	
	prev_k0 = np.zeros(k0.shape)
	while True:
		b_theta_arr = []
		for i in range(1, Nf+1):
			dfilter = directional_filter((i*180)/Nf, 31)
			b_theta = apply_filter(b0, dfilter)
			b_theta_arr.append(b_theta)
		b_theta_arr = np.array(b_theta_arr)

		k_theta_arr = []
		r_transforms = []
		for i in range(0, Nf):
			k_theta = compute_ktheta(b_theta_arr[i], l0, k0)
			r_transform = randon_projection(k_theta, [angle])
			r_transforms.append(r_transform)

		r_transforms = np.array(r_transforms)
		k0, _ = inverse_radon_reconstruction(r_transforms, theta_arr)

		l0 = compute_l_zero(b0, l0, k0)

		if np.linalg.norm(prev_k0-k0) < 0.05:
			break

	return k0, l0


if __name__ == "__main__":
	blur_img = imread('test.jpg', as_gray=True)
	k0 = k_estimation(blur_img)





