import torch
from skimage.io import imread, imsave
import pdb
import numpy as np
import pickle
# import torch.nn.functional as F
import matplotlib.pyplot as plt


def compute_ktheta(b_theta, latent_img, k_init, learning_rate=0.0005, verbose=False):
    """Value for latent image and kernel."""
    latent_img = np.reshape(latent_img, (1, 1, latent_img.shape[0], latent_img.shape[1]))
    latent_img = torch.from_numpy(latent_img)
    latent_img = latent_img.type('torch.FloatTensor')
    latent_img.requires_grad = False

    b_theta = np.reshape(b_theta, (1, 1, b_theta.shape[0], b_theta.shape[1]))
    b_theta = torch.from_numpy(b_theta)
    b_theta = b_theta.type('torch.FloatTensor')
    b_theta.requires_grad = False

    conv_ktheta = torch.nn.Conv2d(1, 1, kernel_size=31, stride=1, padding=15, bias=False)
    conv_ktheta.weight = torch.nn.Parameter(torch.from_numpy(k_init).float().unsqueeze(0).unsqueeze(0))

    # sobel filter
    s_x = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    conv_x = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = torch.nn.Parameter(torch.from_numpy(s_x).float().unsqueeze(0).unsqueeze(0))

    s_y = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    conv_y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = torch.nn.Parameter(torch.from_numpy(s_y).float().unsqueeze(0).unsqueeze(0))

    for param in conv_x.parameters():
        param.requires_grad = False
    for param in conv_y.parameters():
        param.requires_grad = False

    B_x = conv_x(b_theta)
    B_y = conv_y(b_theta)
    delta_B = torch.sqrt(torch.pow(B_x, 2) + torch.pow(B_y, 2))

    L_x = conv_x(latent_img)
    L_y = conv_y(latent_img)
    delta_L = torch.sqrt(torch.pow(L_x, 2) + torch.pow(L_y, 2))

    if torch.cuda.device_count() > 0:
        delta_L = delta_L.cuda()
        delta_B = delta_B.cuda()
        conv_ktheta = conv_ktheta.cuda()

    normval = np.inf
    i = 0
    optimizer = torch.optim.Adam([conv_ktheta.weight], lr=0.005)
    while True:
        # Minimizing Kernel
        out = conv_ktheta(delta_L)

        norm = torch.norm((delta_B - out), 2)
        G = torch.norm(conv_ktheta.weight, 2)

        energy = norm + G
        optimizer.zero_grad()
        energy.backward()

        optimizer.step()

        if i % 100 == 0 and verbose:
            print('Iteration ', i, "Norm = ", norm.item())
        i += 1

        if normval - norm.item() < 0.001:
            break
        normval = norm.item()

    conv_weight = conv_ktheta.weight.detach()
    kernel = conv_weight.cpu().numpy()[0][0]

    return kernel


if __name__ == "__main__":
    img_name = "test.jpg"

    blur_img = imread('test.jpg', as_gray=True)
    L, K = pickle.load(open(img_name.split(".")[0] + "_init.pkl", "rb"))
    # blur_img = rescale(blur_img, 1.0/2, multichannel=False, )

    k_theta = compute_ktheta(blur_img, L, K, verbose=True)
    pdb.set_trace()

    imsave(img_name.split(".")[0] + "L0.png", L)
    imsave(img_name.split(".")[0] + "K0.png", K)
